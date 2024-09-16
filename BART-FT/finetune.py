import argparse
import os.path
import time
from collections import OrderedDict
import torch
from tqdm import tqdm
import data_loader
from transformers import AutoTokenizer
from rouge_util import cal_rouge
from FtModel import FtModel


def train(args, model, tokenizer, train_set, val_set, test_set):
    train_date = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
    start_time = time.time()
    model.train()

    optimizer = torch.optim.Adam(model.model.parameters(), lr=args.learning_rate)
    train_len = len(train_set)
    best_val_rl = 0
    no_improve = 0
    for e in range(args.epoch):
        train_set.shuffle_data()
        print('Epoch [{}/{}]'.format(e + 1, args.epoch))
        total_loss = 0
        for i in tqdm(range(train_len)):
            src_ids, tgt_ids, src_mask, tgt_mask, labels = train_set.__next__()
            optimizer.zero_grad()
            loss = model(src_ids, src_mask, tgt_ids, labels=labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        rg = evaluate(args, model, tokenizer, val_set)
        val_rl = rg[2]
        if val_rl > best_val_rl:
            best_val_rl = val_rl
            model_path = os.path.join(args.model_save_path, 'best_model_{}.pt'.format(train_date))
            torch.save(model.state_dict(), model_path)
            args.model_path = model_path
            improve = '*'
            no_improve = 0
        else:
            improve = ''
            no_improve += 1
        use_time = time.time() - start_time
        print('Epoch: {}, Val set score: [R1:{:.4f},R2:{:.4f},Rl:{:.4f}], loss: {:.4f}, time: {}s{}'.format(e + 1, rg[0], rg[1], rg[2], total_loss / train_len, use_time, improve))
        model.train()
        if no_improve >= args.early_stop:
            print('Early stop training!')
            break
    print('-'*50)
    print('Train finished, start test')
    best_model_test(args, model, tokenizer, test_set)


def evaluate(args, model, tokenizer, val_set):
    model.eval()
    predict_all = []
    target_all = []
    with torch.no_grad():
        for src_ids, tgt_ids, src_mask, tgt_mask, labels in val_set:
            generate_ids = model.model.generate(input_ids=src_ids, attention_mask=src_mask,
                                                max_length=args.max_tgt_length, num_beams=args.num_beams,
                                                early_stopping=True)
            pred = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                    generate_ids]
            tgt = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                   tgt_ids]

            predict_all.extend(pred)
            target_all.extend(tgt)

    score = cal_rouge(predict_all, target_all)
    scores = [score['rouge-1']['f'], score['rouge-2']['f'], score['rouge-l']['f']]
    return scores


def best_model_test(args, model, tokenizer, test_set):
    state_dict = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(state_dict)
    model.eval()
    predict_all = []
    target_all = []
    with torch.no_grad():
        for src_ids, tgt_ids, src_mask, tgt_mask, labels in tqdm(test_set):
            generate_ids = model.model.generate(input_ids=src_ids, attention_mask=src_mask,
                                                max_length=args.max_tgt_length, num_beams=args.num_beams,
                                                early_stopping=True)
            pred = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                    generate_ids]
            tgt = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                   tgt_ids]
            predict_all.extend(pred)
            target_all.extend(tgt)

    score = cal_rouge(predict_all, target_all)
    print('Test set score: R1:{:.4f},R2:{:.4f},Rl:{:.4f}'.format(score['rouge-1']['f'], score['rouge-2']['f'], score['rouge-l']['f']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finetune Medical Question Summarization Base Model")
    parser.add_argument("--dataset", type=str, default="MeQSum", choices=["MeQSum", "CHQ-Summ", "iCliniq", "HealthCareMagic"])
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cuda", default=0)
    parser.add_argument("--model_path", type=str, help="checkpoint position")
    parser.add_argument("--mod", type=str, default="train")
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_src_length", type=float, default=128)
    parser.add_argument("--max_tgt_length", type=float, default=20)
    parser.add_argument("--model_save_path", type=str, default="model/MeQSum", help="path to save the checkpoint")
    parser.add_argument("--early_stop", type=int, default=5)

    args = parser.parse_args()
    device = 'cuda:{}'.format(args.cuda)
    args.device = device

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

    train_set, val_set, test_set = data_loader.load_dataset(os.path.join('dataset/', args.dataset))
    train_set = data_loader.DatasetIterator(tokenizer, train_set, args.batch_size, device, args.max_src_length)
    val_set = data_loader.DatasetIterator(tokenizer, val_set, args.batch_size, device, args.max_src_length)
    test_set = data_loader.DatasetIterator(tokenizer, test_set, args.batch_size, device, args.max_src_length)

    model = FtModel()
    model = model.to(device)

    if args.mod == 'train':
        train(args, model, tokenizer, train_set, val_set, test_set)
    elif args.mod == 'test':
        best_model_test(args, model, tokenizer, test_set)
