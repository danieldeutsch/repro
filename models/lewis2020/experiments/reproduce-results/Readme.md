## Description
This experiment runs prediction using the models on the CNN/DailyMail or XSum dataset and evaluates the predictions using ROUGE to compare to the results reported in the original paper.

## Usage
```
sh models/lewis2020/experiments/reproduce-results/run.sh
```
After prediction is done, we calculated the ROUGE scores via SacreROUGE.
This code is not included in the script because SacreROUGE is not a dependency right now.
```
sacrerouge rouge evaluate \
    --input-files <path/to/predictions.jsonl> \
    --macro-output-json macro.json \
    --micro-output-jsonl micro.jsonl \
    --dataset-reader reference-based \
    --compute_rouge_l true
```
It is necessary to sentence-split the CNN/DailyMail predictions in order to reproduce the ROUGE-L score.
We did this with [`sentence_split.py`](sentence_split.py) and `spacy==2.3.3` and `en_core_web_sm==2.3.1`.

## Results
CNN/DailyMail
||R1|R2|RL|
|-|-|-|-|
|Reported|44.16|21.28|40.90|
|Ours|44.31|21.12|41.23|

XSum
||R1|R2|RL|
|-|-|-|-|
|Reported|45.14|22.27|37.25|
|Ours|44.56|20.93|35.34|
The differences for XSum seems to be [a known issue](https://github.com/pytorch/fairseq/issues/1971). 