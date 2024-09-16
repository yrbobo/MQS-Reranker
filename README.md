# Improving Medical Question Summarization through Re-ranking
This is the code repository for the paper "Improving Medical Question Summarization through Re-ranking".
## Folder Introduction

| Folder           | Intro                                           |
| ---------------- | ----------------------------------------------- |
| BART-FT          | Fine-tuning the BART model.                     |
| SimGate Reranker | Using SimGate Reranker as a second-stage model. |
| LLM Reranker     | Using a large language model as a reranker.     |
## Requirements
Python >= 3.8

pytorch == 1.10.1

transformers == 4.26.1

rouge==1.0.1

py-rouge == 1.1

## Dataset
The datasets can be downloaded from the following URLs.

| Dataset         | URLs                                                         |
| --------------- | ------------------------------------------------------------ |
| MeQSum          | https://github.com/abachaa/MeQSum                            |
| CHQ-Summ        | https://github.com/shwetanlp/Yahoo-CHQ-Summ                  |
| iCliniq         | https://drive.google.com/drive/u/1/folders/1FQTsgRYDJajcNlKJXG-FFPKFw4Cf4FzU |
| HealthCareMagic | https://drive.google.com/drive/u/1/folders/1Hq4AiYr96jfOsB8OJMlyDRRUhmr_BYvY |
## Steps to Run

### 1. Fine-tuning the BART model

```shell
python finetune.py --dataset MeQSum --epoch 20 --model_save_path model/MeQSum
```

### 2. Generating candidate summaries

```shell
python candidate_generation.py --dataset MeQSum --set train --model_path model.MeQSum/your_model.pt --num_return_sequences 16
```

### 3. Train SimGate Reranker

```shell
python ReRankingMain.py --dataset MeQSum --epoch 50 --margin 0.01 --model_save_path reranking_model/MeQSum --mod train
```

### 4. Test SimGate Reranker

```shell
python ReRankingMain.py --dataset MeQSum --model_path reranking_model/MeQSum/your_model.pt --gate_threshold 0.1 
```

### 5. Test LLM as Reranker

```shell
python ChatGLMasReRanker.py --dataset MeQSum --num_cand 4
python LLaMAasReRanker.py --dataset MeQSum --numcand 4 --model LLaMA-vicuna-7B/LLaMA-vicuna-13B
```

