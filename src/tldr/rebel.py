import os
import random
import time
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import wandb
import deepspeed
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import gather_object
from datasets import load_dataset
from rich.console import Console
from rich.pretty import pprint
from rich.table import Table
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
)
from peft import get_peft_model, LoraConfig


@dataclass
class AdaptiveKLParams:
    target: float = 6.0
    horizon: int = 10000  # in episodes


@dataclass
class RewardHParams:
    use_adaptive_kl: bool = False
    adaptive_kl: Optional[AdaptiveKLParams] = field(default_factory=AdaptiveKLParams)
    kl_coef: float = 0.05


@dataclass
class REBELHParams:
    num_updates: tyro.conf.Suppress[int] = 1000
    noptepochs: int = 4
    whiten_rewards: bool = False
    shift_mean: bool = False
    eta: float = 1.0


@dataclass
class TaskHParams:
    # Query params
    query_length: int = 512
    query_dataset: str = "cleanrl/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1704563162" # pythia 2.9

    # Response params
    response_length: int = 53

    # Truncate response after the first occurrence of this token at or after index after when sampling.
    truncate_token: Literal["eos"] = "eos"
    truncate_token_id: Optional[int] = None
    penalty_reward_value: int = -1

    # LM params
    temperature: float = 0.7


@dataclass
class Args:
    # common args
    exp_name: str = "pythia_rebel"
    """the name of this experiment"""
    seed: int = 555134
    """seed of the experiment"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "tldr_summarize_pythia"
    """the wandb's project name"""
    cuda: bool = True
    """Whether to use cuda if available."""
    run_name: Optional[str] = None
    """a unique name of this run"""
    push_to_hub: bool = False
    "whether to upload the saved model to huggingface"
    hf_entity: str = ""
    "the user or org name of the model repository from the Hugging Face Hub"
    deepspeed: bool = True
    """Whether to use deepspeed to train the model"""
    print_sample_output_freq: int = 100
    """How often to print sample output"""
    run_eval: bool = True
    """Whether to run evaluation"""

    # optimizer args
    eps: float = 1e-5
    """the epsilon value for the optimizer"""
    lr: float = 3e-6
    """the learning rate"""
    optimizer: Literal["adam", "adamw"] = "adamw"
    """Which optimizer to use"""

    gradient_accumulation_steps: int = 8
    """The number of gradient accumulation steps"""
    per_device_train_batch_size: int = 8
    """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""
    per_device_eval_batch_size: int = 2
    """per rank eval batch size"""
    total_episodes: int = 1000000
    """The total number of episodes in the dataset"""

    # optional args filled while running
    world_size: Optional[int] = 4
    """The number of processes (GPUs) to use"""
    batch_size: Optional[int] = 512
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    local_rollout_forward_batch_size: int = 32
    """per rank no grad forward pass in the rollout phase"""
    local_batch_size: Optional[int] = 128
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""

    # other args
    base_model: str = "models/sft_tldr_pythia_1_4b"
    """the name of the pretrained model to use"""
    offload: bool = False
    """Whether to offload ref policy and reward model to CPU"""
    reward_model_path: str = "models/rm_sft_tldr_pythia_1_4b"
    """the name of the pretrained model to use"""
    sft_model_path: str = "models/sft_tldr_pythia_1_4b"
    """the name of the pretrained model to use"""
    dropout_layer_keys: List[str] = field(
        default_factory=lambda: ["attn_pdrop", "embd_pdrop", "resid_pdrop", "summary_first_dropout"]
    )
    """Which layers to apply dropout to"""
    output_dir: str = "models/rebel_tldr_pythia_1_4b"
    """Where to save the model"""
    lora_rank: int = 1024
    """the rank of the lora matrix"""
    lora_alpha: int = 2048
    """weight of lora"""
    lora_dropout: float = 0.0
    """dropout for lora"""
    task: TaskHParams = field(default_factory=TaskHParams)
    reward: RewardHParams = field(default_factory=RewardHParams)
    rebel: REBELHParams = field(default_factory=REBELHParams)


# taken from https://github.com/microsoft/DeepSpeedExamples/blob/737c6740bec38b77a24a59135b6481a53d566b38/applications/DeepSpeed-Chat/training/utils/model/model_utils.py#L20C1-L26C52
def configure_dropout(model_config, dropout_layer_keys, dropout):
    if dropout is not None:
        for key in dropout_layer_keys:
            if hasattr(model_config, key):
                print(f"Setting model_config.{key} to {dropout}")
                setattr(model_config, key, dropout)


def print_rich_table(title: str, df: pd.DataFrame, console: Console) -> Table:
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.rule(f"[bold red]{title}")
    console.print(table)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer


class AdaptiveKLController:
    def __init__(self, init_kl_coef: float, hparams: AdaptiveKLParams):
        self.value = init_kl_coef
        self.hparams = hparams

    def update(self, current, n_steps):
        target = self.hparams.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.hparams.horizon
        self.value *= mult


def whiten(values, shift_mean=True):
    mean, var = torch.mean(values), torch.var(values, unbiased=False)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


class ScalarModelConfig(PretrainedConfig):
    def __init__(
        self,
        base_model: str = "EleutherAI/pythia-160m",
        base_config: PretrainedConfig = AutoConfig.from_pretrained("EleutherAI/pythia-160m"),
        hidden_size: int = 768,
        bias: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.base_config = base_config
        self.hidden_size = hidden_size
        self.bias = bias


class ScalarModel(PreTrainedModel):
    config_class = ScalarModelConfig

    def __init__(self, config: ScalarModelConfig):
        super().__init__(config)
        self.config = config
        self.lm_backbone = AutoModel.from_pretrained(
            config.base_model,
            config=self.config.base_config,
            trust_remote_code=True,
        )
        self.scalar_head = layer_init(
            nn.Linear(self.config.hidden_size, 1),
            std=1 / np.sqrt(self.config.hidden_size + 1),
        )

    def forward(self, **kwargs):
        output = self.lm_backbone(**kwargs)
        reward = self.scalar_head(output.hidden_states[-1]) - self.config.bias
        return reward


def get_reward(model, query_responses, tokenizer, context_length):
    attention_mask = query_responses != tokenizer.pad_token_id
    # position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    reward_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )
    sequence_lengths = first_true_indices(query_responses[:, context_length:] == tokenizer.pad_token_id) - 1 + context_length
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    return (
        reward_logits,
        reward_logits[torch.arange(reward_logits.size(0), device=reward_logits.device), sequence_lengths].squeeze(-1),
        sequence_lengths,
    )


def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError(f"Inexact division: {a} / {b} = {a / b}")
    return q


def generate(lm_backbone, queries, tokenizer, generation_config):
    """generate in a way that does not affect padding tokens"""
    context_length = queries.shape[1]
    attention_mask = queries != tokenizer.pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # generation collapsed if this was turned on. TODO: why does generation collapse with this?
        generation_config=generation_config,
        return_dict_in_generate=True,
    )
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1)


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def truncate_response(args, tokenizer, responses):
    trunc_idxs = first_true_indices(responses == args.task.truncate_token_id).unsqueeze(-1)
    new_size = [1] * (len(responses.size()) - 1) + [args.task.response_length]
    idxs = torch.arange(args.task.response_length, device=responses.device).view(*new_size)
    postprocessed_responses = torch.masked_fill(responses, idxs > trunc_idxs, tokenizer.pad_token_id)
    return postprocessed_responses


def forward(model, query_responses, tokenizer, ref=False):
    attention_mask = query_responses != tokenizer.pad_token_id
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    if ref:
        with model.disable_adapter():
            return model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
    else:
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )


@dataclass
class EvalStorage:
    query_token: List[str] = field(default_factory=list)
    postprocessed_response_token: List[str] = field(default_factory=list)
    reference_response_token: List[str] = field(default_factory=list)
    score: List[float] = field(default_factory=list)
    reference_score: List[float] = field(default_factory=list)

    query: List[str] = field(default_factory=list)
    postprocessed_response: List[str] = field(default_factory=list)
    reference_response: List[str] = field(default_factory=list)
    kl: List[float] = field(default_factory=list)


def evaluate(args: Args, reward_model, policy, tokenizer, dataloader, generation_config, sampling=True):
    eval_storage = EvalStorage()
    with torch.no_grad():
        for data in tqdm(dataloader):
            queries = data["query_token"]
            reference_response_token = data["reference_response_token"]
            context_length = queries.shape[1]
            query_reference_responses = torch.cat((data["query_token"], data["reference_response_token"]), dim=1)
            _, reference_score, _ = get_reward(reward_model, query_reference_responses, tokenizer, queries.shape[1])

            query_responses = generate(
                policy,
                queries,
                tokenizer,
                generation_config,
            )
            responses = query_responses[:, context_length:]

            output = forward(policy, query_responses, tokenizer)
            logits = output.logits[:, context_length - 1 : -1]
            logits /= generation_config.temperature
            all_logprob = F.log_softmax(logits, dim=-1)
            logprobs = torch.gather(all_logprob, 2, responses.unsqueeze(-1)).squeeze(-1)
            del output, logits, all_logprob
            torch.cuda.empty_cache()

            ref_output = forward(policy, query_responses, tokenizer, ref=True)
            ref_logits = ref_output.logits[:, context_length - 1 : -1]
            ref_logits /= generation_config.temperature
            ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
            ref_logprobs = torch.gather(ref_all_logprob, 2, responses.unsqueeze(-1)).squeeze(-1)
            del ref_output, ref_logits, ref_all_logprob
            torch.cuda.empty_cache()
            
            postprocessed_responses = truncate_response(args, tokenizer, responses)
            postprocessed_query_responses = torch.cat((queries, postprocessed_responses), 1)
            _, score, _ = get_reward(reward_model, postprocessed_query_responses, tokenizer, queries.shape[1])
            kl = (logprobs - ref_logprobs).sum(1)

            eval_storage.query_token.extend(queries)
            eval_storage.reference_response_token.extend(reference_response_token)
            eval_storage.reference_score.append(reference_score)
            eval_storage.postprocessed_response_token.extend(postprocessed_responses)
            eval_storage.score.append(score)
            eval_storage.kl.append(kl)
            
            if sampling:
                break

    eval_storage.query = tokenizer.batch_decode(eval_storage.query_token, skip_special_tokens=True)
    eval_storage.reference_response = tokenizer.batch_decode(eval_storage.reference_response_token)
    eval_storage.postprocessed_response = tokenizer.batch_decode(
        eval_storage.postprocessed_response_token, skip_special_tokens=True
    )
    eval_score = torch.cat(eval_storage.score).float().cpu().numpy().tolist()
    eval_reference_score = torch.cat(eval_storage.reference_score).float().cpu().numpy().tolist()
    eval_kl = torch.cat(eval_storage.kl).float().cpu().numpy().tolist()
    eval_df = pd.DataFrame(
        {
            "query": gather_object(eval_storage.query),
            "postprocessed_response": gather_object(eval_storage.postprocessed_response),
            "reference_responses": gather_object(eval_storage.reference_response),
            "scores": gather_object(eval_score),
            "reference_scores": gather_object(eval_reference_score),
            "kl": gather_object(eval_kl),
        }
    )
    return eval_storage, eval_df


if __name__ == "__main__":

    args = tyro.cli(Args)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    local_seed = args.seed + accelerator.process_index * 100003  # Prime

    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)

    args.world_size = accelerator.num_processes
    args.batch_size = args.per_device_train_batch_size * args.world_size * args.gradient_accumulation_steps
    args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    if args.rebel.whiten_rewards:
        assert (args.local_batch_size >= 8), f"Per-rank minibatch size {args.local_batch_size} is insufficient for whitening"
    args.rebel.num_updates = args.total_episodes // args.batch_size

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
        trust_remote_code=True,
    )
    # we use the padding token manually but do not resize the token embedding of the model
    if args.task.truncate_token == "eos":
        args.task.truncate_token_id = tokenizer.eos_token_id
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    console = Console(force_terminal=True)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}__{args.output_dir.split('/')[1]}"
    print("Wandb run name: ", run_name)
    writer = SimpleNamespace()  # dummy writer
    writer.add_scalar = lambda x, y, z: None
    writer.add_histogram = lambda x, y, z, max_bins: None
    if accelerator.is_main_process:
        if args.track:
            wandb.init(
                project=args.wandb_project_name,
                sync_tensorboard=True,
                config=asdict(args),
                name=run_name,
                save_code=True,
            )
            file_extensions = [".toml", ".lock", ".py", ".sh", ".yaml"]
            wandb.run.log_code(".", include_fn=lambda path: any([path.endswith(ext) for ext in file_extensions]))
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        pprint(args)
    device = accelerator.device
    torch.backends.cudnn.deterministic = True

    model_config = AutoConfig.from_pretrained(args.base_model)
    configure_dropout(model_config, args.dropout_layer_keys, 0.0)  # disable dropout
    scalar_model_config = ScalarModelConfig(
        base_model=args.base_model,
        base_config=model_config,
        hidden_size=model_config.hidden_size,
    )
    if not args.reward_model_path:
        reward_model: PreTrainedModel = ScalarModel(scalar_model_config)
    else:
        reward_model: PreTrainedModel = ScalarModel.from_pretrained(
            args.reward_model_path,
            trust_remote_code=True,
        )
    if accelerator.is_main_process:
        pprint(model_config)
        pprint(reward_model.config)

    policy = AutoModelForCausalLM.from_pretrained(args.sft_model_path, config=model_config, trust_remote_code=True)
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    policy = get_peft_model(policy, peft_config=peft_config)
    accelerator.print(policy)
    policy.generation_config.eos_token_id = None  # disable `pad_token_id` and `eos_token_id` because we just want to
    policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding
    
    if args.optimizer == "adam":
        optimizer = optim.Adam(policy.parameters(), lr=args.lr, eps=args.eps)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(policy.parameters(), lr=args.lr, eps=args.eps)

    dataset = load_dataset(args.task.query_dataset, split="train")
    dataset = dataset.with_format("torch", columns=["query_token", "reference_response_token"])
    dataloader = DataLoader(dataset, batch_size=args.local_batch_size, shuffle=True)
    validation_dataset = load_dataset(args.task.query_dataset, split="validation")
    validation_dataset = validation_dataset.with_format("torch", columns=["query_token", "reference_response_token"])
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.per_device_eval_batch_size)

    # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
    # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
    torch.manual_seed(args.seed)
    policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)
    validation_dataloader = accelerator.prepare(validation_dataloader)
    def repeat_generator():
        while True:
            yield from dataloader
    iter_dataloader = iter(repeat_generator())
    torch.manual_seed(local_seed)  # reset the local seed again

    if args.deepspeed:
        deepspeed_states = AcceleratorState().deepspeed_plugin
        deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size

        eval_ds_config = {
            "train_micro_batch_size_per_gpu": deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"],
            "bf16": {"enabled": True},
            "prescale_gradients": False,
            "wall_clock_breakdown": False,
        }
        if args.offload:
            deepspeed_states.deepspeed_config["checkpoint"] = {"use_node_local_storage": True}
            eval_ds_config["zero_optimization"] = {
                "stage": 3,
                "stage3_param_persistence_threshold": 1e4,
                "offload_param": {"device": "cpu"},
            }
        accelerator.print(f"{eval_ds_config=}")
        reward_model, *_ = deepspeed.initialize(model=reward_model, config=eval_ds_config)
        reward_model.eval()
    else:
        reward_model = reward_model.to(device)

    kl_ctl = AdaptiveKLController(args.reward.kl_coef, hparams=args.reward.adaptive_kl)
    # WARNING: even with `max_new_tokens` and `min_new_tokens` set to the same value, the number of tokens generated
    # may not be the same. TODO: investigate further, we just want to generate a fixed number of tokens
    generation_config = GenerationConfig(
        max_new_tokens=args.task.response_length,
        min_new_tokens=args.task.response_length,
        temperature=(args.task.temperature + 1e-7),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )
    # use the same `0.01` temperature for validation response generation https://github.com/openai/summarize-from-feedback/blob/700967448d10004279f138666442bf1497d0e705/exps/sample.py#L27
    validation_generation_config = GenerationConfig(
        max_new_tokens=args.task.response_length,
        min_new_tokens=args.task.response_length,
        temperature=(0.01 + 1e-7),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )

    accelerator.print("===training policy===")
    global_step = 0
    start_time = time.time()
    stats_shape = (args.rebel.noptepochs, args.gradient_accumulation_steps)

    approxkl_stats = torch.zeros(stats_shape, device=device)
    loss_stats = torch.zeros((args.rebel.noptepochs, args.gradient_accumulation_steps), device=device)
    entropy_stats = torch.zeros(stats_shape, device=device)
    ratio_stats = torch.zeros(stats_shape, device=device)

    policy.train()
    for update in range(1, args.rebel.num_updates + 1):
        global_step += 1 * args.batch_size
        frac = 1.0 - (update - 1.0) / args.rebel.num_updates
        lrnow = frac * args.lr
        optimizer.param_groups[0]["lr"] = lrnow
        data = next(iter_dataloader)
        with torch.no_grad():
            print("sampling evaluation")
            eval_storage, eval_df = evaluate(
                args,
                reward_model,
                accelerator.unwrap_model(policy),
                tokenizer,
                validation_dataloader,
                validation_generation_config,
            )
            validation_score = eval_storage.score[0]
            if args.print_sample_output_freq > 0 and update > 1 and (update - 1) % args.print_sample_output_freq == 0:
                if accelerator.is_main_process:
                    eval_df.to_csv(f"runs/{run_name}/table_{global_step}.csv")
                    if args.track:
                        wandb.log({"samples/query_responses": wandb.Table(dataframe=eval_df)}, step=update)
                    else:
                        try:
                            print_rich_table(f"Sample Output at Step {update}", eval_df[:1], console)
                        except Exception as e:
                            print(e)
                if args.run_eval:
                    eval_storage, eval_df = evaluate(
                        args,
                        reward_model,
                        accelerator.unwrap_model(policy),
                        tokenizer,
                        validation_dataloader,
                        validation_generation_config,
                        sampling=False,
                    )
                    if accelerator.is_main_process:
                        eval_df.to_csv(f"runs/{run_name}/table.csv")
                        if args.track:
                            wandb.log({f"eval/query_responses_{update}": wandb.Table(dataframe=eval_df)}, step=update)

                # save model
                if args.output_dir:
                    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
                    time_tensor = torch.tensor([int(time.time())], device=device)
                    time_int = accelerator.gather(time_tensor)[0].item()  # avoid different timestamps across processes
                    repo_name = f"{args.base_model.replace('/', '_')}__{args.exp_name}__tldr"
                    repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name

                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(args.output_dir)
                        if args.push_to_hub:
                            tokenizer.push_to_hub(repo_id, revision=f"seed{args.seed}_{str(time_int)}")

                    unwrapped: PreTrainedModel = accelerator.unwrap_model(policy)
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        unwrapped.save_pretrained(
                            args.output_dir,
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                            state_dict=accelerator.get_state_dict(unwrapped),
                            safe_serialization=False,
                            repo_id=repo_id,
                        )
                        if args.push_to_hub:
                            unwrapped.push_to_hub(repo_id, revision=f"seed{args.seed}_{str(time_int)}", safe_serialization=False)
            del eval_storage, eval_df
            torch.cuda.empty_cache()

            print("generating rollouts")
            queries = data["query_token"].to(device)
            context_length = queries.shape[1]

            query_responses = []
            responses = []
            postprocessed_responses = []
            logprobs = []
            ref_logprobs = []
            scores = []
            sequence_lengths = []

            for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                query = queries[i : i + args.local_rollout_forward_batch_size]

                batch_query_responses = []
                batch_responses = []
                batch_postprocessed_responses = []
                batch_logprobs = []
                batch_ref_logprobs = []
                batch_scores = []
                batch_sequence_lengths = []

                for _ in range(2):
                    query_response = generate(
                        accelerator.unwrap_model(policy),
                        query,
                        tokenizer,
                        generation_config,
                    )
                    response = query_response[:, context_length:]

                    output = forward(accelerator.unwrap_model(policy), query_response, tokenizer)
                    logits = output.logits[:, context_length - 1 : -1]
                    logits /= args.task.temperature + 1e-7
                    all_logprob = F.log_softmax(logits, dim=-1)
                    logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                    del output, logits, all_logprob
                    torch.cuda.empty_cache()

                    ref_output = forward(accelerator.unwrap_model(policy), query_response, tokenizer, ref=True)
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits /= args.task.temperature + 1e-7
                    ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                    ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                    del ref_output, ref_logits, ref_all_logprob
                    torch.cuda.empty_cache()

                    # Response Processing 1. truncate response after the first occurrence of `truncate_token_id`
                    postprocessed_response = truncate_response(args, tokenizer, response)

                    # Response Processing 2. run reward model on the truncated responses
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    sequence_length = first_true_indices(postprocessed_response == tokenizer.pad_token_id) - 1
                    _, score, _ = get_reward(reward_model, postprocessed_query_response, tokenizer, context_length)

                    batch_query_responses.append(query_response)
                    batch_responses.append(response)
                    batch_postprocessed_responses.append(postprocessed_response)
                    batch_logprobs.append(logprob)
                    batch_ref_logprobs.append(ref_logprob)
                    batch_scores.append(score)
                    batch_sequence_lengths.append(sequence_length)

                query_responses.append(torch.stack(batch_query_responses, 1))
                responses.append(torch.stack(batch_responses, 1))
                postprocessed_responses.append(torch.stack(batch_postprocessed_responses, 1))
                logprobs.append(torch.stack(batch_logprobs, 1))
                ref_logprobs.append(torch.stack(batch_ref_logprobs, 1))
                scores.append(torch.stack(batch_scores, 1))
                sequence_lengths.append(torch.stack(batch_sequence_lengths, 1))

            query_responses = torch.cat(query_responses, 0).flatten(end_dim=1)
            responses = torch.cat(responses, 0).flatten(end_dim=1)
            postprocessed_responses = torch.cat(postprocessed_responses, 0).flatten(end_dim=1)
            logprobs = torch.cat(logprobs, 0).flatten(end_dim=1)
            ref_logprobs = torch.cat(ref_logprobs, 0).flatten(end_dim=1)
            scores = torch.cat(scores, 0).flatten(end_dim=1)
            sequence_lengths = torch.cat(sequence_lengths, 0).flatten(end_dim=1)
            del (logprob, ref_logprob, score, batch_query_responses, batch_responses, batch_postprocessed_responses, \
                 batch_logprobs, batch_ref_logprobs, batch_scores, batch_sequence_lengths)
            torch.cuda.empty_cache()

            # Response Processing 3. filter response. Ensure that the sample contains truncate_token_id
            # responses not passing that filter will receive a low (fixed) score
            # only query humans on responses that pass that filter
            contain_pad_token = torch.any(postprocessed_responses == tokenizer.pad_token_id, dim=-1)
            scores = torch.where(contain_pad_token, scores, torch.full_like(scores, args.task.penalty_reward_value))
            accelerator.print(f"{scores=}, {(contain_pad_token.sum() / len(contain_pad_token))=}")

            # 4. cumulative logprob
            seq_mask = torch.arange(responses.size(1), device=policy.device).unsqueeze(0).expand_as(responses) <= sequence_lengths.unsqueeze(1)
            logprobs = (logprobs * seq_mask).sum(-1)
            ref_logprobs = (ref_logprobs * seq_mask).sum(-1)

            # 5. kl reward and normalization
            kl = logprobs - ref_logprobs
            non_score_reward = -kl_ctl.value * kl
            rewards = non_score_reward + scores
            if args.rebel.whiten_rewards:
                rewards = whiten(rewards, args.rebel.shift_mean)

            accelerator.print("rewards with kl====", rewards)
            if accelerator.is_main_process:
                console.print(
                    f"mean_kl",
                    kl.mean().item(),
                    "scores",
                    scores.mean().item(),
                )
            torch.cuda.empty_cache()

        # Do multiple epochs of REBEL training, with a fresh random shuffle in each epoch
        for rebel_epoch_idx in range(args.rebel.noptepochs):
            local_batch_idxs = np.random.permutation(args.local_batch_size)
            gradient_accumulation_idx = 0
            for mini_batch_start in range(0, args.local_batch_size, args.per_device_train_batch_size):
                mini_batch_end = mini_batch_start + args.per_device_train_batch_size
                mini_batch_inds = local_batch_idxs[mini_batch_start:mini_batch_end] * 2
                mini_batch_inds = np.append(mini_batch_inds, mini_batch_inds + 1)
                with accelerator.accumulate(policy):
                    mb_responses = responses[mini_batch_inds]
                    mb_query_responses = query_responses[mini_batch_inds]
                    mb_logprobs = logprobs[mini_batch_inds]
                    mb_rewards = rewards[mini_batch_inds]
                    mb_seq_mask = seq_mask[mini_batch_inds]

                    output = forward(policy, mb_query_responses, tokenizer)
                    logits = output.logits[:, context_length - 1 : -1]
                    logits /= args.task.temperature + 1e-7
                    new_all_logprobs = F.log_softmax(logits, dim=-1)
                    new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                    new_logprobs = (new_logprobs * mb_seq_mask).sum(-1)

                    ratio_logprob = new_logprobs - mb_logprobs
                    ratio_logprob = ratio_logprob[:args.per_device_train_batch_size] - ratio_logprob[args.per_device_train_batch_size:]
                    reg_diff = ratio_logprob - args.rebel.eta * (mb_rewards[:args.per_device_train_batch_size] - mb_rewards[args.per_device_train_batch_size:])
                    loss = (reg_diff ** 2).mean()

                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                    with torch.no_grad():
                        y = args.rebel.eta * (mb_rewards[:args.per_device_train_batch_size] - mb_rewards[args.per_device_train_batch_size:])
                        logprobs_diff = new_logprobs - mb_logprobs
                        ratio = torch.exp(logprobs_diff)
                        prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                        approxkl = 0.5 * (logprobs_diff**2).mean()
                        approxkl_stats[rebel_epoch_idx, gradient_accumulation_idx] = approxkl
                        loss_stats[rebel_epoch_idx, gradient_accumulation_idx] = loss
                        entropy_stats[rebel_epoch_idx, gradient_accumulation_idx] = entropy.mean()
                        ratio_stats[rebel_epoch_idx, gradient_accumulation_idx] = ratio.mean()
                gradient_accumulation_idx += 1
            if accelerator.is_main_process:
                console.print(
                    f"rebel_epoch_idx",
                    rebel_epoch_idx,
                    "approxkl",
                    approxkl_stats[rebel_epoch_idx].mean().item(),
                    "loss",
                    loss_stats[rebel_epoch_idx].mean().item(),
                )
                
        with torch.no_grad():
            mean_kl = kl.mean()
            mean_entropy = -logprobs.mean()
            mean_non_score_reward = non_score_reward.mean()
            writer.add_scalar("objective/kl_coef", kl_ctl.value, update)
            writer.add_scalar("objective/kl", accelerator.gather(mean_kl).mean().item(), update)
            writer.add_scalar("objective/entropy", accelerator.gather(mean_entropy).mean().item(), update)
            writer.add_scalar("objective/non_score_reward", accelerator.gather(mean_non_score_reward).mean().item(), update)
            writer.add_scalar("objective/score_total", accelerator.gather(mean_non_score_reward + scores.mean()).mean().item(), update)
            writer.add_scalar("objective/scores", accelerator.gather(scores.mean()).mean().item(), update)
            writer.add_scalar("objective/validation_score", accelerator.gather(validation_score.mean()).mean().item(), update)
            writer.add_histogram("objective/scores_his", accelerator.gather(scores).cpu().float().numpy().flatten(), update, max_bins=64)
            writer.add_histogram("objective/validation_scores_his", accelerator.gather(validation_score).cpu().float().numpy().flatten(), update, max_bins=64)
            writer.add_scalar("rebel/loss/policy", accelerator.gather(loss).mean().item(), update)
            writer.add_scalar("rebel/policy/entropy", accelerator.gather(entropy.mean()).mean().item(), update)
            writer.add_scalar("rebel/policy/approxkl", accelerator.gather(approxkl).mean().item(), update)

            writer.add_scalar("rebel/policy/initial_loss", accelerator.gather(loss_stats[0]).mean().item(), update)
            writer.add_scalar("rebel/policy/final_loss", accelerator.gather(loss_stats[-1]).mean().item(), update)
            writer.add_scalar("rebel/policy/delta_loss", accelerator.gather(loss_stats[-1] - loss_stats[0]).mean().item(), update)
            
            writer.add_scalar("npg/policy/approxkl_avg", accelerator.gather(approxkl_stats).mean().item(), update)
            writer.add_scalar("npg/loss/policy_avg", accelerator.gather(loss_stats).mean().item(), update)
            writer.add_scalar("npg/policy/entropy_avg", accelerator.gather(entropy_stats).mean().item(), update)
            writer.add_scalar("npg/val/ratio", accelerator.gather(ratio_stats).mean().item(), update)
            writer.add_scalar("npg/val/ratio_var", accelerator.gather(ratio_stats).var().item(), update)
            writer.add_scalar("npg/val/num_eos_tokens", (responses == tokenizer.eos_token_id).sum().item(), update)
            writer.add_scalar("npg/lr", lrnow, update)
            writer.add_scalar("npg/episode", global_step, update)
            eps = int(global_step / (time.time() - start_time))
            writer.add_scalar("npg/eps", eps, update)
            accelerator.print("npg/eps", eps, update)
            if args.reward.use_adaptive_kl:
                kl_ctl.update(mean_kl.item(), args.batch_size)
            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores
            torch.cuda.empty_cache()

    if args.run_eval:
        eval_storage, eval_df = evaluate(
            args,
            reward_model,
            accelerator.unwrap_model(policy),
            tokenizer,
            validation_dataloader,
            validation_generation_config,
            sampling=False,
        )
        if accelerator.is_main_process:
            eval_df.to_csv(f"runs/{run_name}/table.csv")
            if args.track:
                wandb.log({"eval/query_responses": wandb.Table(dataframe=eval_df)}, step=update)

    # save model
    if args.output_dir:
        os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
        time_tensor = torch.tensor([int(time.time())], device=device)
        time_int = accelerator.gather(time_tensor)[0].item()  # avoid different timestamps across processes
        repo_name = f"{args.base_model.replace('/', '_')}__{args.exp_name}__tldr"
        repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name

        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir, repo_id=repo_id)
            if args.push_to_hub:
                tokenizer.push_to_hub(repo_id, revision=f"seed{args.seed}_{str(time_int)}")

        unwrapped: PreTrainedModel = accelerator.unwrap_model(policy)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(unwrapped),
                safe_serialization=False,
                repo_id=repo_id,
            )
            if args.push_to_hub:
                unwrapped.push_to_hub(repo_id, revision=f"seed{args.seed}_{str(time_int)}", safe_serialization=False)
