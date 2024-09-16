import argparse
import json
import os.path
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import data_loader
from FtModel import FtModel
from rouge_util import cal_rouge
import stanza


def get_summaries(args, tokenizer, dataset, dataset_raw, model):
    texts = []
    summaries = []
    labels = []
    model = model.model

    for src_ids, tgt_ids, src_mask, tgt_mask, label in tqdm(dataset):
        generate_ids = model.generate(input_ids=src_ids,
                                      attention_mask=src_mask,
                                      max_length=args.max_tgt_length,
                                      num_beams=args.num_beams,
                                      num_return_sequences=args.num_return_sequences,
                                      early_stopping=True)
        src = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
               src_ids]
        pred = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                generate_ids]
        tgt = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
               tgt_ids]
        summaries += pred
        texts += src
        labels += tgt

    return texts, summaries, labels


def cal_best_score(summaries, labels):
    print(len(labels), len(summaries))
    candidate_num = len(summaries) // len(labels)
    R1_best, R2_best, RL_best = 0.0, 0.0, 0.0
    candidate_objs = []
    for idx, label in tqdm(enumerate(labels)):
        best_score = [0, 0, 0]
        for candidate in summaries[idx * candidate_num:idx * candidate_num + candidate_num]:
            rg_score = cal_rouge(candidate, label)
            r1, r2, rl = rg_score['rouge-1']['f'], rg_score['rouge-2']['f'], rg_score['rouge-l']['f']
            candidate_obj = Candidate(candidate, rl, 0)
            candidate_objs.append(candidate_obj)
            if rl > best_score[2]:
                best_score = [r1, r2, rl]
        R1_best += best_score[0]
        R2_best += best_score[1]
        RL_best += best_score[2]
    R1_best /= len(labels)
    R2_best /= len(labels)
    RL_best /= len(labels)
    return R1_best, R2_best, RL_best, candidate_objs


def cal_score(summaries, labels):
    candidate_num = len(summaries) // len(labels)
    R1, R2, RL = 0.0, 0.0, 0.0
    for idx, label in enumerate(labels):
        for candidate in summaries[idx * candidate_num:idx * candidate_num + candidate_num]:
            rg_score = cal_rouge(candidate, label)
            r1, r2, rl = rg_score['rouge-1']['f'], rg_score['rouge-2']['f'], rg_score['rouge-l']['f']
            R1 += r1
            R2 += r2
            RL += rl
            break
    R1 /= len(labels)
    R2 /= len(labels)
    RL /= len(labels)
    return R1, R2, RL


class Candidate:
    def __init__(self, summary, rouge_score, ent_score):
        self.summary = summary
        self.candidate_rouge_score = rouge_score
        self.candidate_ent_score = ent_score

    def __str__(self):
        return 'summary: {}, candidate_rouge_score: {}, candidate_ent_score: {}'.format(self.summary, self.candidate_rouge_score, self.candidate_ent_score)


class Data:
    def __init__(self, src, candidate, tgt):
        self.src = src
        self.candidate = candidate
        self.tgt = tgt


if __name__ == '__main__':
    parser = argparse.ArgumentParser('generate summary candidates')
    parser.add_argument('--dataset', type=str, default='MeQSum')
    parser.add_argument('--set', type=str, default='val')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--model_path', type=str,
                        default='/home/ubuntu/MedQA/MedSumma/model/MeQSum/best_model_2023-06-24-11-11.pt')
    parser.add_argument('--num_beams', type=int, default=16)
    parser.add_argument('--generation_method', type=str, default='beam_search')
    parser.add_argument('--max_src_length', type=float, default=128)
    parser.add_argument('--max_tgt_length', type=float, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_return_sequences', type=int, default=16)

    args = parser.parse_args()
    device = 'cuda:{}'.format(args.cuda)
    args.device = device
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

    train_set_raw, val_set_raw, test_set_raw = data_loader.load_dataset(os.path.join('dataset/', args.dataset))
    train_set = data_loader.DatasetIterator(tokenizer, train_set_raw, args.batch_size, device, args.max_src_length)
    val_set = data_loader.DatasetIterator(tokenizer, val_set_raw, args.batch_size, device, args.max_src_length)
    test_set = data_loader.DatasetIterator(tokenizer, test_set_raw, args.batch_size, device, args.max_src_length)

    model = FtModel()
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()
    print('Loaded the model weights!', args.model_path)
    print('dataset: {}, set: {}'.format(args.dataset, args.set))
    if args.set == 'train':
        texts, summaries, labels = get_summaries(args, tokenizer, train_set, train_set_raw, model)
        r1, r2, rl, candidates = cal_best_score(summaries, labels)
        candidate_num = len(summaries) // len(labels)
        for idx, (text, label) in enumerate(zip(texts, labels)):
            single_json_file = open("reranking_dataset/{}/{}/{}.json".format(args.dataset, args.set, idx), 'w',
                                    encoding='utf-8')
            candidate_obj_list = []
            for candidate in candidates[idx * candidate_num:idx * candidate_num + candidate_num]:
                candidate_obj = [''.join(candidate.summary), candidate.candidate_rouge_score]
                candidate_obj_list.append(candidate_obj)
            json_obj = {
                "chq": text,
                "faq": label,
                "candidates": candidate_obj_list
            }
            json_str = json.dumps(json_obj)
            single_json_file.write(json_str.strip() + '\n')
            single_json_file.close()
    elif args.set == 'val':
        texts, summaries, labels = get_summaries(args, tokenizer, val_set, val_set_raw, model)
        r1, r2, rl, candidates = cal_best_score(summaries, labels)
        candidate_num = len(summaries) // len(labels)
        for idx, (text, label) in enumerate(zip(texts, labels)):
            single_json_file = open("reranking_dataset/{}/{}/{}.json".format(args.dataset, args.set, idx), 'w',
                                    encoding='utf-8')
            candidate_obj_list = []
            for candidate in candidates[idx * candidate_num:idx * candidate_num + candidate_num]:
                candidate_obj = [''.join(candidate.summary), candidate.candidate_rouge_score]
                candidate_obj_list.append(candidate_obj)
            json_obj = {
                "chq": text,
                "faq": label,
                "candidates": candidate_obj_list
            }
            json_str = json.dumps(json_obj)
            single_json_file.write(json_str.strip() + '\n')
            single_json_file.close()
    elif args.set == 'test':
        texts, summaries, labels = get_summaries(args, tokenizer, test_set, test_set_raw, model)
        print(cal_score(summaries, labels))
        r1, r2, rl, candidates = cal_best_score(summaries, labels)
        candidate_num = len(summaries) // len(labels)
        for idx, (text, label) in enumerate(zip(texts, labels)):
            single_json_file = open("reranking_dataset/{}/{}/{}.json".format(args.dataset, args.set, idx), 'w',
                                    encoding='utf-8')
            candidate_obj_list = []
            for candidate in candidates[idx * candidate_num:idx * candidate_num + candidate_num]:
                candidate_obj = [''.join(candidate.summary), candidate.candidate_rouge_score]
                candidate_obj_list.append(candidate_obj)
            json_obj = {
                "chq": text,
                "faq": label,
                "candidates": candidate_obj_list
            }
            json_str = json.dumps(json_obj)
            single_json_file.write(json_str.strip() + '\n')
            single_json_file.close()

