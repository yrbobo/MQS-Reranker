import argparse
import json
import os
import string
import time
from fastchat.conversation import get_conv_template
from rouge_util import Rouge_py_rouge
from fastchat.serve.cli import SimpleChatIO
from inference import generate_stream
from fastchat.model.model_adapter import load_model, get_conversation_template


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
    if len(answer) == 1:
        idx = letter_to_num(answer)
        if 0 <= idx <= 15:
            return candidates[idx][0]
        else:
            return candidates[0][0]
    else:
        for candidate in candidates:
            cand = candidate[0]
            if cand in answer:
                return cand
        return candidates[0][0]


def main(args):
    print(args)
    now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    model_type = '/home/ubuntu/MedQA/LLM/LLaMA-vicuna/{}'.format(args.model)
    chatio = SimpleChatIO()
    chatio.prompt_for_output('ASSISTANT')
    model, tokenizer = load_model(model_type, 'cuda', 1, None, False, False, False)
    print('LLaMA-Vicuna as ReRanker START!')
    cnt = 0
    r1, r2, rl = 0, 0, 0
    train_set_len = len(os.listdir(f'dataset/{args.dataset}/train'))
    conv = get_conversation_template(model_type)
    for idx in range(0, train_set_len):
        with open(f'dataset/{args.dataset}/train/{idx}.json', 'r') as f:
            data = json.load(f)
            chq = data['chq']
            candidates = data['candidates']
            query = 'This is a multiple choice. \n' \
                    'Question: ' \
                    'Which option best expresses the intention of the following paragraph: ' \
                    f'{chq}\n{build_choices_str(candidates, args.num_cand)}\n ' \
                    f'Answer: '
            ans = 'A'
            score = candidates[0][1]
            print(candidates)
            for k in range(1, args.num_cand):
                if candidates[k][1] > score:
                    score = candidates[k][1]
                    ans = num_to_letter(k + 1)
            print(ans)
            conv.append_message(conv.roles[0], query)
            conv.append_message(conv.roles[1], ans)

    if not os.path.exists(f'dataset/{args.dataset}/LLM_result'):
        os.makedirs(f'dataset/{args.dataset}/LLM_result')
        print(f"Folder created successfully.")
    else:
        print(f"Folder already exists.")

    with open(f'dataset/{args.dataset}/LLM_result/{args.model}_answers_{args.num_cand}_{now_time}_{args.do_sample}.txt', 'w') as ans:
        test_set_len = len(os.listdir(f'dataset/{args.dataset}/test'))
        for idx in range(0, test_set_len):
            with open(f'dataset/{args.dataset}/test/{idx}.json', 'r') as f:
                data = json.load(f)
                chq = data['chq']
                candidates = data['candidates']
                query = 'This is a multiple choice. \n' \
                        'Question: ' \
                        'Which option best expresses the intention of the following paragraph: ' \
                        f'{chq}\n{build_choices_str(candidates, args.num_cand)}\n ' \
                        f'Answer: '
                prompt_conv = conv.copy()
                prompt_conv.append_message(conv.roles[0], query)
                prompt_conv.append_message(conv.roles[1], None)

                gen_params = {
                    "model": model_type,
                    "prompt": prompt_conv.get_prompt(),
                    "do_sample": args.do_sample,
                    "temperature": 0.7,
                    "stop": None,
                    "stop_token_ids": None,
                    "echo": False,
                }
                output_stream = generate_stream(model, tokenizer, gen_params, 'cuda')
                outputs = chatio.stream_output(output_stream)
                answer = outputs.replace('\n', '')
                cand = get_final_cand(candidates, answer)
                score = Rouge_py_rouge(cand, data['faq'])
                r1 += score['rouge-1']['f']
                r2 += score['rouge-2']['f']
                rl += score['rouge-l']['f']
                print('R1: {}, R2: {}, RL: {}'.format(score['rouge-1']['f'], score['rouge-2']['f'],
                                                      score['rouge-l']['f']))
                cnt += 1
                print(str(cnt) + ' ' + answer)
                ans.write(f'{cand} | {data["faq"]}' + '\n')
    result = open(f'dataset/{args.dataset}/LLM_result/{args.model}_result_{args.num_cand}_{now_time}_{args.do_sample}.txt', 'w')
    result.write('R1: {}, R2: {}, RL: {}'.format(r1 / test_set_len, r2 / test_set_len,
                                                 rl / test_set_len))
    result.close()
    print('R1: {}, R2: {}, RL: {}'.format(r1 / test_set_len, r2 / test_set_len,
                                          rl / test_set_len))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("LLM as ReRanker")
    parser.add_argument("--dataset", type=str, default="MeQSum")
    parser.add_argument("--num_cand", type=int, default="4")
    parser.add_argument("--model", type=str, default="LLaMA-vicuna-7B")
    parser.add_argument("--do_sample", action="store_true")
    args = parser.parse_args()
    main(args)
