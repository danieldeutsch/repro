## Description
This experiment runs prediction using all of the models on the CNN/DailyMail or XSum dataset and evaluates the predictions using ROUGE to compare to the results reported in the original paper.
After running inference, the script also tokenizes the reference summaries to replicate the summaries the authors used for evaluation.

## Usage
```
sh models/liu2019/experiments/reproduce-results/run.sh
```
After prediction is done, we calculated the ROUGE scores via SacreROUGE.
This code is not included in the script because SacreROUGE is not a dependency right now.
```
sacrerouge rouge evaluate \
    --input-files <path/to/predictions-tokenized.jsonl> \
    --macro-output-json macro.json \
    --micro-output-jsonl micro.jsonl \
    --dataset-reader reference-based \
    --compute_rouge_l true
```

## Results
`TransformerAbs` on CNN/DailyMail
||R1|R2|RL|
|-|-|-|-|
|Reported|40.21|17.76|37.09|
|Ours|40.38|17.81|37.10|

`BertSumExt` on CNN/DailyMail
||R1|R2|RL|
|-|-|-|-|
|Reported|43.23|20.24|39.63|
|Ours|41.93|18.98|38.07|

`BertSumExtAbs` on CNN/DailyMail
||R1|R2|RL|
|-|-|-|-|
|Reported|42.13|19.60|39.18|
|Ours|42.08|19.43|38.95|

`BertSumExtAbs` on XSum
||R1|R2|RL|
|-|-|-|-|
|Reported|38.81|16.50|31.27|
|Ours|38.88|16.41|31.31|