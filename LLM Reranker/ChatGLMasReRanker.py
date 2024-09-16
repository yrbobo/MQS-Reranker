import argparse
import json
import os
import string
import time
from transformers import AutoTokenizer, AutoModel
from rouge_util import Rouge_py_rouge

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).cuda()
model = model.eval()


def num_to_letter(num):
    return string.ascii_uppercase[num - 1]


def build_choices_str(candidates, num_cand):
    choices_str = ''
    for i in range(1, num_cand + 1):
        choices_str += f'{num_to_letter(i)}. {candidates[i - 1][0]} \n'
    return choices_str


def letter_to_num(letter):
    return ord(letter) - 65


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
    past_key_values, history = None, []
    now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    print(now_time + 'ChatGLM2-6B as ReRanker START!')
    cnt = 0
    r1, r2, rl = 0, 0, 0
    train_set_len = len(os.listdir(f'dataset/{args.dataset}/train'))
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
                    ans = num_to_letter(k+1)
            print(ans)
            history.append((query, ans))

    if not os.path.exists(f'dataset/{args.dataset}/LLM_result'):
        os.makedirs(f'dataset/{args.dataset}/LLM_result')
        print(f"Folder created successfully.")
    else:
        print(f"Folder already exists.")

    with open(f'dataset/{args.dataset}/LLM_result/ChatGLM2-6B_answers_{args.num_cand}_{now_time}_{args.do_sample}.txt', 'w') as ans:
        test_set_len = len(os.listdir(f'dataset/{args.dataset}/test'))
        for idx in range(0, test_set_len):
            with open(f'dataset/{args.dataset}/test/{idx}.json', 'r') as f:
                data = json.load(f)
                chq = data['chq']
                candidates = data['candidates']
                query = 'This is a multiple choice.\n' \
                        'Question: ' \
                        'Which option best expresses the intention of the following paragraph: ' \
                        f'{chq} \n{build_choices_str(candidates, args.num_cand)}\n' \
                        f'Answer: '
                answer = ''
                for response, his in model.stream_chat(tokenizer, query, history=history, do_sample=args.do_sample):
                    answer = response

                answer = answer.replace('\n', '')
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
    result = open(f'dataset/{args.dataset}/LLM_result/ChatGLM2-6B_result_{args.num_cand}_{now_time}_{args.do_sample}.txt', 'w')
    result.write('R1: {}, R2: {}, RL: {}'.format(r1 / test_set_len, r2 / test_set_len,
                                                 rl / test_set_len))
    result.close()
    print('R1: {}, R2: {}, RL: {}'.format(r1 / test_set_len, r2 / test_set_len,
                                          rl / test_set_len))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("LLM as ReRanker")
    parser.add_argument("--dataset", type=str, default="MeQSum")
    parser.add_argument("--num_cand", type=int, default="4")
    parser.add_argument("--do_sample", action="store_true")
    args = parser.parse_args()
    main(args)
