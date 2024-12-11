import argparse
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List

torch.set_printoptions(threshold=10_000)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--input_repo", type=str, required=True, help="output repo from rank.py")
    parser.add_argument("--maxlen", type=int, default=2048)
    parser.add_argument("--maxlen_prompt", type=int, default=1024)
    parser.add_argument("--pairs", type=int, default=5)
    return parser.parse_args()


def get_message(instruction=None, response=None):

    assert instruction != None or response != None

    if response == None:
        message = [
            {"role": "user", "content": instruction},
        ]
    elif instruction == None:
        message = [
            {"role": "assistant", "content": response}
        ]
    else:
        message = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ]

    return message


def filter_same_responses(row):
    return row['chosen'] != row['reject']


def main():

    # init
    args = parse_arguments()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer_left = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    tokenizer_left.add_special_tokens({"pad_token": "[PAD]"})

    dataset = load_dataset(args.input_repo, split='train')
    
    # process dataset
    print('initial length:', len(dataset))

    # filter dataset with long prompt or response
    dataset = dataset.filter(lambda row: tokenizer.apply_chat_template(get_message(row['prompt']), tokenize=True, add_generation_prompt=True, return_tensors='pt').shape[-1] <= args.maxlen_prompt)
    print('filtered long prompts:', len(dataset))
    for i in range(args.pairs):
        dataset = dataset.filter(lambda row: tokenizer.apply_chat_template(get_message(response=row[f'response_{i}']), tokenize=True, add_generation_prompt=False, return_tensors='pt')[:, 5:].shape[-1] <= args.maxlen)
        print(f'filtered response_{i}:', len(dataset))

    # add prompt tokens
    llama_prompts = []
    llama_prompt_tokens = []
    for row in tqdm(dataset):
        llama_prompt_token = tokenizer_left.apply_chat_template(
                get_message(row['prompt']), 
                add_generation_prompt=True,
                tokenize=True,
                padding='max_length',
                max_length=args.maxlen_prompt,
        )
        llama_prompt = tokenizer_left.decode(llama_prompt_token, skip_special_tokens=False)
        assert len(llama_prompt_token) == args.maxlen_prompt
        assert (llama_prompt_token[0] == 128000 or llama_prompt_token[0] == 128256) and llama_prompt_token[-1] == 271
        llama_prompts.append(llama_prompt)
        llama_prompt_tokens.append(llama_prompt_token)
    dataset = dataset.add_column("llama_prompt", llama_prompts)
    dataset = dataset.add_column("llama_prompt_tokens", llama_prompt_tokens)

    # select chosen and reject
    chosen, reject, llama_chosen, llama_reject, llama_chosen_tokens, llama_reject_tokens, chosen_reward, reject_reward = [], [], [], [], [], [], [], []
    
    for row in dataset:

        all_rewards = [row[f"response_{i}_reward"] for i in range(args.pairs)]
        chosen_idx, reject_idx = np.argmax(all_rewards), np.argmin(all_rewards)

        chosen.append(row[f"response_{chosen_idx}"])
        reject.append(row[f"response_{reject_idx}"])

        llama_chosen_token = tokenizer.apply_chat_template(
                get_message(response=row[f"response_{chosen_idx}"]),
                add_generation_prompt=False,
                tokenize=True,
                padding='max_length',
                max_length=args.maxlen+5,
        )[5:]
        llama_chosen_tokens.append(llama_chosen_token)
        llama_chosen.append(tokenizer.decode(llama_chosen_token, skip_special_tokens=False))
        chosen_reward.append(row[f"response_{chosen_idx}_reward"])
        assert len(llama_chosen_token) == args.maxlen
        assert llama_chosen_token[-1] == 128009 or llama_chosen_token[-1] == 128256

        llama_reject_token = tokenizer.apply_chat_template(
                get_message(response=row[f"response_{reject_idx}"]),
                add_generation_prompt=False,
                tokenize=True,
                padding='max_length',
                max_length=args.maxlen+5,
        )[5:]
        llama_reject_tokens.append(llama_reject_token)
        llama_reject.append(tokenizer.decode(llama_reject_token, skip_special_tokens=False))
        reject_reward.append(row[f"response_{reject_idx}_reward"])
        assert len(llama_reject_token) == args.maxlen
        assert llama_reject_token[-1] == 128009 or llama_reject_token[-1] == 128256

    dataset = dataset.add_column("chosen", chosen)
    dataset = dataset.add_column("chosen_reward", chosen_reward)
    dataset = dataset.add_column("llama_chosen", llama_chosen)
    dataset = dataset.add_column("llama_chosen_tokens", llama_chosen_tokens)
    dataset = dataset.add_column("reject", reject)
    dataset = dataset.add_column("reject_reward", reject_reward)
    dataset = dataset.add_column("llama_reject", llama_reject)
    dataset = dataset.add_column("llama_reject_tokens", llama_reject_tokens)

    # filter prompts with exactly same responses
    dataset = dataset.filter(lambda row: filter_same_responses(row))
    print('filtered same responses:', len(dataset))

    dataset = dataset.train_test_split(test_size=1000, shuffle=True)
    dataset.push_to_hub(args.input_repo + '_tokenized')


if __name__ == "__main__":
    main()