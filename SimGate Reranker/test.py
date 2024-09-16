import json
import os
import torch
from transformers import AutoModel, AutoTokenizer
from RankModel import ReRanker
from rouge_util import cal_rouge

dataset = 'HealthCareMagic'
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = ReRanker('roberta-base', tokenizer.pad_token_id)
model = model.cuda()
model.load_state_dict(torch.load(f'reranking_model/{dataset}/best_margin0.001_2023-06-27-20-16.pt', map_location='cuda:0'))
test_set_len = len(os.listdir(f'reranking_dataset/{dataset}/test'))
refs = []
hyps = []
gate = 0.005
for idx in range(0, test_set_len):
    with open(f'reranking_dataset/{dataset}/test/{idx}.json', 'r') as f:
        texts = []
        data = json.load(f)
        chq = data['chq']
        faq = data['faq']
        texts.append(chq)
        for candidate in data['candidates']:
            texts.append(candidate[0])
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs.to('cuda:0')
        # Get the embeddings
        with torch.no_grad():
            embeddings = model.encoder(**inputs, output_hidden_states=True, return_dict=True).pooler_output

        # Calculate cosine similarities
        # Cosine similarities are in [-1, 1]. Higher means more similar
        doc_emb = embeddings[0]
        cand_emb = embeddings[1:]
        doc_emb = doc_emb.expand_as(cand_emb)
        score = torch.cosine_similarity(doc_emb, cand_emb, dim=-1)

        cos_similarity = score.cpu().numpy()
        print(cos_similarity)
        max_id = cos_similarity.argmax(0)
        if cos_similarity[max_id] - cos_similarity[0] > gate:
            result = data['candidates'][max_id][0]
        else:
            result = data['candidates'][0][0]

        hyps.append(result)
        refs.append(faq)

rg_score = cal_rouge(hyps, refs)
rouge1, rouge2, rougeLsum = rg_score['rouge-1']['f'], rg_score['rouge-2']['f'], rg_score['rouge-l']['f']
print(f"rouge-1: {rouge1}, rouge-2: {rouge2}, rouge-L: {rougeLsum}")


