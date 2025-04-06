# Improving Medical Question Summarization through Re-ranking
This is the code repository for the paper "Improving Medical Question Summarization through Re-ranking".
## Folder Introduction

| Folder           | Intro                                           |
| ---------------- | ----------------------------------------------- |
| BART-FT          | Fine-tuning the BART model.                     |
| SimGate Reranker | Using SimGate Reranker as a second-stage model. |
| LLM Reranker     | Using a large language model as a reranker.     |
## Requirements
### BART/Sim Gate Reranker
Python >= 3.8

pytorch == 1.10.1

transformers == 4.26.1

rouge==1.0.1

py-rouge == 1.1
### LLM Reranker
Python >= 3.8

pytorch == 2.0.1

transformers == 4.42.2 (GLM4)

transformers == 4.46.0 (Llama3.1/Qwen2.5)
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
python llm_reranker_main.py --dataset MeQSum --dataset_path dataset/MeQSum/test/test.json --cand_num 4 --model Meta-Llama-3.1-8B-Instruct --model_path models/Meta-Llama-3.1-8B-Instruct --example_num 3 --log_path log
```

## Acknowledgement

If this work is useful in your research, please cite our paper.

```
@inproceedings{wei2025improving,
  title={Improving Medical Question Summarization through Re-ranking},
  author={Wei, Sibo and Peng, Xueping and Jiang, Yan and Li, Zhao and Liu, Yan and Wang, Zhiqiang and Lu, Wenpeng},
  booktitle={Proceedings of the 2025 International Joint Conference on Neural Network (IJCNN 2025)},
  pages={1--8},
  year={2025}
}
```

