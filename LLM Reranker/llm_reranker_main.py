import argparse
from tqdm import tqdm
from rouge_util import cal_rouge
import json
import torch
import string
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import LogTool
import os
import logging

def load_dataset(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def num_to_letter(num):
    return string.ascii_uppercase[num - 1]

def letter_to_num(letter):
    return ord(letter) - 65

def build_choices_str(candidates, num_cand):
    choices_str = ''
    for i in range(1, num_cand + 1):
        choices_str += f'{num_to_letter(i)}. {candidates[i - 1][0]} \n'
    return choices_str

def get_final_cand(candidates, answer):
    answer = answer.replace('\n', '')
    if len(answer) == 1:
        idx = letter_to_num(answer)
        if 0 <= idx <= 15:
            return candidates[idx][0]
        else:
            return candidates[0][0]
    else:
        answer = answer.lower()
        for candidate in candidates:
            cand = candidate[0].lower()
            if cand in answer:
                return cand
        return candidates[0][0]

def message_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message["role"] == "user":
            prompt += f'{message["content"]}'
        else:
            prompt += f'{message["content"]}\n'
    return prompt

def build_prompt(data):
    prompt = f'Answer the following question. Please provide only one answer option, no additional information is required.\n' \
            f'Question: Which option is the best summary of the following content?\n' \
            f'Content: {data["chq"].strip()}\n' \
            f'Option: {build_choices_str(data["candidates"], args.cand_num)}' \
            f'Answer: '

    return prompt

def build_history_messages(args):
    example_datas = load_dataset(args.dataset_path.replace('test', 'val'))[:args.example_num]
    history_messages = []
    for example_data in example_datas:
        prompt = build_prompt(example_data)
        candidates = example_data['candidates']
        option = 'A'
        # ans = f'{option}. {candidates[0][0]}'
        score = candidates[0][1]
        for i in range(1, args.cand_num):
            if candidates[i][1] > score:
                score = candidates[i][1]
                option = num_to_letter(i + 1)
                # ans = f'{option}. {candidates[i][0]}'
        history_messages.append({"role": "user", "content": prompt})
        history_messages.append({"role": "assistant", "content": option})

    return history_messages

def run(args, logger):
    if 'Llama' in args.model_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map=args.device,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            trust_remote_code=True
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    datas = load_dataset(args.dataset_path)
    history_messages = build_history_messages(args)

    rouge_1, rouge_2, rouge_L = 0, 0, 0
    count = 0
    for data in tqdm(datas):
        prompt = build_prompt(data)
        messages = [_ for _ in history_messages]
        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        resp = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        pred = get_final_cand(data['candidates'], resp)
        ref = data['faq']
        rg_score = cal_rouge([pred], [ref])
        r1, r2, rL = rg_score['rouge-1']['f'], rg_score['rouge-2']['f'], rg_score['rouge-l']['f']
        rouge_1 += r1
        rouge_2 += r2
        rouge_L += rL
        count += 1
        logger.info(f'Pred: {resp}. {pred} || Ref: {ref}')
        logger.info(f'Current Average Score: [ROUGE-1, {rouge_1/count:.4f}], [ROUGE-2, {rouge_2/count:.4f}], [ROUGE-L, {rouge_L/count:.4f}]')

    logger.info(f'Average Score: [ROUGE-1, {rouge_1/count:.4f}], [ROUGE-2, {rouge_2/count:.4f}], [ROUGE-L, {rouge_L/count:.4f}]')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MQS with LLM Reranker")
    parser.add_argument("--dataset", type=str, default='CHQ-Summ')
    parser.add_argument("--dataset_path", type=str, default='CHQ-Summ/test/test.json')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--cand_num", type=int, default=4)
    parser.add_argument("--model", type=str, default='Meta-Llama-3.1-8B-Instruct')
    parser.add_argument("--model_path", type=str, default='Meta-Llama-3.1-8B-Instruct')
    parser.add_argument("--example_num", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--log_path", type=str, default='log')
    args = parser.parse_args()

    log_name = f"{args.model}_{args.dataset}_cand{args.cand_num}.log"
    log_tool = LogTool(log_file=os.path.join(args.log_path, args.dataset, log_name), log_level=logging.DEBUG)
    logger = log_tool.get_logger()

    run(args, logger)