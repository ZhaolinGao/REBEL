# REBEL

This repo covers the implementation for our paper [REBEL](https://arxiv.org/abs/2404.16767). 

Zhaolin Gao, Jonathan D. Chang, Wenhao Zhan, Owen Oertell, Gokul Swamy, Kianté Brantley, Thorsten Joachims, J. Andrew Bagnell, Jason D. Lee, Wen Sun. "REBEL: Reinforcement Learning via Regressing Relative Rewards"

![front page](./figs/rebel_ffig.png)

## Environment

```
torch>=2.1.0
transformers>=4.34
accelerate>=0.23
peft==0.6.2
bitsandbytes>=0.41.1
deepspeed>=0.10.3
tyro
scipy
rouge
shortuuid
jsonlines
rich
wandb
tensorboard
pandas
evaluate
```

## TL;DR Summarization

#### Supervised Fine-tuning (SFT)

You can train your own SFT model by running:
```
accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml \
                  --main_process_port 29085 --num_processes 8 \
                      src/tldr/sft.py \
                      --base_model EleutherAI/pythia-{SIZE}b-deduped \
                      --output_dir models/sft_tldr_pythia_{SIZE}b
```
Alternatively, you can use the existing 2.8B and 6.9B SFT models:
```
vwxyzjn/EleutherAI_pythia-2.8b-deduped__sft__tldr
vwxyzjn/EleutherAI_pythia-6.9b-deduped__sft__tldr
```

#### Reward Model

You can train your own RM model by running:
```
accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml \
                  --main_process_port 29085 --num_processes 8 \
                      src/tldr/rm.py \
                      --base_model models/sft_tldr_pythia_{SIZE}b
                      --output_dir models/rm_sft_tldr_pythia_{SIZE}b
```
Alternatively, you can use the existing 2.8B and 6.9B RM models:
```
vwxyzjn/EleutherAI_pythia-2.8b-deduped__reward__tldr
vwxyzjn/EleutherAI_pythia-6.9b-deduped__reward__tldr
```

#### REBEL

You can run REBEL by
```
./scripts/tldr/rebel.sh
```

We also include a script for PPO.
```
./scripts/tldr/ppo.sh
```

## General Chat

We apply REBEL on two different sets of models and datasets for general chat.

|  | Base Model | Reward Model | Dataset |
| -------- | ------- |  ------- |  ------- | 
| Set 1 | [OpenChat-3.5](https://huggingface.co/openchat/openchat_3.5)   | [Starling-RM-7B-alpha](https://huggingface.co/berkeley-nest/Starling-RM-7B-alpha) | [Nectar](https://huggingface.co/datasets/berkeley-nest/Nectar) |
| Set 2 | [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | [FsfairX-LLaMA3-RM-v0.1](https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1) | [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) |

Our preprocessed dataset can be found at [Nectar](https://huggingface.co/datasets/jdchang/nectar_openchat_preprocess) and [UltraFeedback](https://huggingface.co/datasets/GitBag/ultrafeedback_llama3_eurus).

You can run REBEL on Nectar and UltraFeedback by
```
./scripts/nectar/rebel.sh
./scripts/ultrafeedback/rebel.sh
```

Below is a list of models that are trained with the scripts above.

| Model | #P | MMLU<br>(5-shot) | MT-Bench<br>1st Turn | MT-Bench<br>2nd Turn | MT-Bench<br>Average | AlpacaEval 2.0<br>LC Win Rate | AlpacaEval 2.0<br>Win Rate |
| -------- | ------- |  ------- | ------- |------- |------- |------- |------- |
| [REBEL-OpenChat-3.5](https://huggingface.co/Cornell-AGI/REBEL-OpenChat-3.5) | 7B | 63.7 | 8.54 | 7.58 | 8.06 | 17.3 | 12.8 |
| [REBEL-Llama-3](https://huggingface.co/Cornell-AGI/REBEL-Llama-3) | 8B | 65.8 | 8.63 | 7.69 | 8.16 | 30.1 | 32.6 |


## RLCM

Please refer to [RLCM](https://github.com/Owen-Oertell/rlcm) repo for the implementation of REBEL on RLCM.

## Cite
Please cite our paper if you use this implementation in your own work:
```
@misc{gao2024rebel,
      title={REBEL: Reinforcement Learning via Regressing Relative Rewards}, 
      author={Zhaolin Gao and Jonathan D. Chang and Wenhao Zhan and Owen Oertell and Gokul Swamy and Kianté Brantley and Thorsten Joachims and J. Andrew Bagnell and Jason D. Lee and Wen Sun},
      year={2024},
      eprint={2404.16767},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgments
Thanks for [summarize_from_feedback_details](https://github.com/vwxyzjn/summarize_from_feedback_details/tree/62c37d63c212c55bde52833611eb642a95facb5c) on which this repository is initially based.
