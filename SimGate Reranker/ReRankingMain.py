import argparse
import json
import os
import random
import time
from functools import partial
from torch import optim, nn
from tqdm import tqdm
from RankModel import ReRanker, RankingLoss
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, BertTokenizer
from ReRankingDataset import to_cuda, collate_mp, ReRankingDataset
from rouge_util import cal_rouge


def train(args, scorer, dataloader, val_dataloader):
    print(f'args: {args}')
    train_date = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
    with open(f'log/{args.dataset}/{args.dataset}_test_log_{train_date}.txt', 'a') as f:
        f.write(f'args: {args}')
    init_lr = args.max_lr / args.warmup_steps
    optimizer = optim.Adam(scorer.parameters(), lr=init_lr)
    all_step_cnt = 0
    minimum_loss = 1000000
    no_improve = 0
    for epoch in range(args.epoch):
        optimizer.zero_grad()
        avg_loss = 0
        step_cnt = 0
        for idx, batch in tqdm(enumerate(dataloader)):
            to_cuda(batch, 0)
            step_cnt += 1
            output = scorer(batch["src_input_ids"], batch["candidate_ids"], batch["tgt_input_ids"])
            similarity, gold_similarity = output['score'], output['summary_score']
            ranking_loss = RankingLoss(similarity, gold_similarity, args.margin, args.gold_margin, args.gold_weight)
            loss = ranking_loss / args.accumulate_step
            avg_loss += loss.item()
            loss.backward()
            if step_cnt == args.accumulate_step:
                if args.grad_norm > 0:
                    nn.utils.clip_grad_norm_(scorer.parameters(), args.grad_norm)
                step_cnt = 0
                all_step_cnt += 1
                lr = args.max_lr * min(all_step_cnt ** (-0.5), all_step_cnt * (args.warmup_steps ** (-1.5)))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                optimizer.step()
                optimizer.zero_grad()
        del similarity, gold_similarity, loss
        print('Epoch {}, loss: {}'.format(epoch, avg_loss / len(dataloader)))

        loss = evaluation(val_dataloader, scorer, args)
        if loss < minimum_loss:
            no_improve = 0
            minimum_loss = loss
            model_path = os.path.join(args.model_save_path, 'best_margin{}_{}.pt'.format(args.margin, train_date))
            torch.save(scorer.state_dict(), model_path)
            args.model_path = model_path
            print('The best reranking model has been updated!')
            t1, t2, t3 = re_ranker_test(args)
            with open(f'log/{args.dataset}/{args.dataset}_test_log_{train_date}.txt', 'a') as f:
                f.write(f"rouge-1: {t1}, rouge-2: {t2}, rouge-L: {t3}\n")
        else:
            no_improve += 1

        if no_improve >= args.early_stop:
            print('early stop!')
            break


def evaluation(dataloader, scorer, args):
    scorer.eval()
    loss = 0
    cnt = 0
    rouge1, rouge2, rougeLsum = 0, 0, 0
    with torch.no_grad():
        for (i, batch) in enumerate(dataloader):
            to_cuda(batch, args.cuda)
            samples = batch["data"]
            output = scorer(batch["src_input_ids"], batch["candidate_ids"], batch["tgt_input_ids"])
            similarity, gold_similarity = output['score'], output['summary_score']
            similarity = similarity.cpu().numpy()
            if i % 100 == 0:
                print(f"test similarity: {similarity[0]}")
            max_ids = similarity.argmax(1)
            for j in range(similarity.shape[0]):
                cnt += 1
                sample = samples[j]
                sents = sample["candidates"][max_ids[j]][0]
                score = cal_rouge(sents, sample["faq"])
                rouge1 += score['rouge-1']['f']
                rouge2 += score['rouge-2']['f']
                rougeLsum += score['rouge-l']['f']
    rouge1 = rouge1 / cnt
    rouge2 = rouge2 / cnt
    rougeLsum = rougeLsum / cnt
    scorer.train()
    loss = 1 - ((rouge1 + rouge2 + rougeLsum) / 3)
    print(f"eval: rouge-1: {rouge1}, rouge-2: {rouge2}, rouge-L: {rougeLsum}")
    return loss


def re_ranker_test(args):
    test_date = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
    tokenizer = RobertaTokenizer.from_pretrained(args.model_type)
    collate_fn_test = partial(collate_mp, pad_token_id=tokenizer.pad_token_id, is_test=True)
    test_set = ReRankingDataset("reranking_dataset/{}/{}".format(args.dataset, 'test'), args.model_type, is_test=True,
                                maxlen=512, is_sorted=False, maxnum=args.max_num)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                 collate_fn=collate_fn_test)
    model_path = args.model_path
    scorer = ReRanker(args.model_type, tokenizer.pad_token_id)
    scorer = scorer.cuda()
    scorer.load_state_dict(torch.load(model_path, map_location='cuda:{}'.format(args.cuda)))
    scorer.eval()
    print("load model: {}".format(model_path))
    hyps = []
    refs = []
    with torch.no_grad():
        for (i, batch) in tqdm(enumerate(test_dataloader)):
            to_cuda(batch, args.cuda)
            samples = batch["data"]
            output = scorer(batch["src_input_ids"], batch["candidate_ids"], batch["tgt_input_ids"])
            similarity, gold_similarity = output['score'], output['summary_score']
            similarity = similarity.cpu().numpy()
            if i % 100 == 0:
                print(f"test similarity: {similarity}")
            max_ids = similarity.argmax(1)
            for j in range(similarity.shape[0]):
                sample = samples[j]
                sents = sample["candidates"][max_ids[j]][0]
                if args.gate_threshold is not None:
                    if similarity[j][max_ids[j]] - similarity[j][0] > args.gate_threshold:
                        hyps.append(sents)
                        refs.append(sample["faq"])
                    else:
                        hyps.append(sample["candidates"][0][0])
                        refs.append(sample["faq"])
                else:
                    hyps.append(sents)
                    refs.append(sample["faq"])

    scorer.train()
    rg_score = cal_rouge(hyps, refs)
    rouge1, rouge2, rougeLsum = rg_score['rouge-1']['f'], rg_score['rouge-2']['f'], rg_score['rouge-l']['f']

    print(f"test: rouge-1: {rouge1}, rouge-2: {rouge2}, rouge-L: {rougeLsum}")
    result_file = open(f'reranking_model/{args.dataset}/test_result_{test_date}.txt', 'w')
    result_file.write(f"rouge-1: {rouge1}, rouge-2: {rouge2}, rouge-L: {rougeLsum}\n")
    result_file.write(f'{args.gate_threshold}')
    result_file.close()
    return rouge1, rouge2, rougeLsum


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ReRanker Training")
    parser.add_argument("--dataset", type=str, default="MeQSum",
                        choices=["MeQSum", "CHQ-Summ", "iCliniq", "HealthCareMagic"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--gold_margin", type=float, default=0)
    parser.add_argument("--cand_weight", type=float, default=1)
    parser.add_argument("--gold_weight", type=float, default=1)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--grad_norm", type=int, default=0)
    parser.add_argument("--max_lr", type=float, default=0.002)
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--seed", type=int, default=970903)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--max_num", type=int, default=16)
    parser.add_argument("--model_path", type=str, help="checkpoint position")
    parser.add_argument("--model_save_path", type=str, default="reranking_model/MeQSum")
    parser.add_argument("--accumulate_step", type=int, default=1)
    parser.add_argument("--early_stop", type=int, default=100)
    parser.add_argument("--mod", type=str, default="train")
    parser.add_argument("--model_type", type=str, default="roberta-base")
    parser.add_argument("--gate_threshold", type=float, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    print("model_type: {}".format(args.model_type))
    if args.mod == "train":
        tokenizer = RobertaTokenizer.from_pretrained(args.model_type)
        collate_fn = partial(collate_mp, pad_token_id=tokenizer.pad_token_id, is_test=False)
        collate_fn_val = partial(collate_mp, pad_token_id=tokenizer.pad_token_id, is_test=True)

        train_set = ReRankingDataset("reranking_dataset/{}/{}".format(args.dataset, 'train'), args.model_type,
                                     maxlen=args.max_len, maxnum=args.max_num)
        val_set = ReRankingDataset("reranking_dataset/{}/{}".format(args.dataset, 'val'), args.model_type, is_test=True,
                                   maxlen=512, is_sorted=False, maxnum=args.max_num)
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                collate_fn=collate_fn)
        val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                    collate_fn=collate_fn_val)

        scorer = ReRanker(args.model_type, tokenizer.pad_token_id)
        scorer = scorer.cuda()
        scorer.train()
        train(args, scorer, dataloader, val_dataloader)
    elif args.mod == "test":
        re_ranker_test(args)
