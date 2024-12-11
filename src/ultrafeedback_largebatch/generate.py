from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import argparse
import torch
import random
import numpy as np


def set_seed(seed=5775709):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--prompts", type=str, default="allenai/ultrafeedback_binarized_cleaned_train")
    parser.add_argument("--output_repo", type=str, required=True, help="output repo for the generated reponses")
    parser.add_argument("--maxlen", type=int, default=2048)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--world_size", type=int, default=4)
    return parser.parse_args()


def get_message(instruction):
    message = [
        {"role": "user", "content": instruction},
    ]
    return message


def main():

    # init
    args = parse_arguments()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.world_size,
    )

    # dataset
    dataset = load_dataset(args.prompts, split='train')
    if args.end_idx != -1:
        dataset = dataset.select(range(args.start_idx, args.end_idx))

    # prompts for llm
    prompts = [tokenizer.apply_chat_template(get_message(row['prompt']), tokenize=False, add_generation_prompt=True) for row in tqdm(dataset)]

    # start generate
    for p in range(args.pairs):
        set_seed(p * 50)
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.9,
            max_tokens=args.maxlen,
            seed=p * 50,
        )
        response = llm.generate(prompts, sampling_params)
        output = list(map(lambda x: x.outputs[0].text, response))
        dataset = dataset.add_column(f"response_{p}", output)

    # clean and push
    columns = ["prompt_id", "prompt"] + [f"response_{i}" for i in range(args.pairs)]
    dataset = dataset.select_columns(columns)
    dataset.push_to_hub(args.output_repo)


if __name__ == "__main__":
    main()