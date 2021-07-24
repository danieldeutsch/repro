## Description
This experiment runs the oracle sentence-guided model and evaluates its predictions with ROUGE to compare to what is reported in the paper.

## Usage
```
sh models/dou2021/experiments/reproduce-results/run.sh
```

After prediction is done, we calculated the ROUGE scores via SacreROUGE.
This code is not included in the script because SacreROUGE is not a dependency right now.
It is necessary to sentence-split the CNN/DailyMail predictions in order to reproduce the ROUGE-L score.
We did this with [`sentence_split.py`](sentence_split.py) and `spacy==2.3.3` and `en_core_web_sm==2.3.1`.
```shell script
python sentence_split.py --input-file predictions.jsonl --output-file predictions-split.jsonl

sacrerouge rouge evaluate \
    --input-files <path/to/predictions-tokenized.jsonl> \
    --macro-output-json macro.json \
    --micro-output-jsonl micro.jsonl \
    --dataset-reader reference-based \
    --compute_rouge_l true
```

## Results
CNN/DailyMail - Oracle Sentence-Guided
||R1|R2|RL|
|-|-|-|-|
|Reported|55.18|32.54|52.06|
|Ours|52.36|29.22|49.13|

CNN/DailyMail - Sentence-Guided Model with BertSumExt
||R1|R2|RL|
|-|-|-|-|
|Reported|43.78|20.66|40.66|
|Ours||||