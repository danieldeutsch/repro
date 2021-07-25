## Description
This experiment runs prediction using the models on the CNN/DailyMail or XSum dataset and evaluates the predictions using ROUGE to compare to the results reported in the original paper.

## Usage
Running this experiments requires having ROUGE setup (see [here](../../../sacrerouge/Readme.md)).
```shell script
sh models/lewis2020/experiments/reproduce-results/run.sh
```

## Results
CNN/DailyMail
||R1|R2|RL|
|-|-|-|-|
|Reported|44.16|21.28|40.90|
|Ours|44.31|21.12|41.18|

XSum
||R1|R2|RL|
|-|-|-|-|
|Reported|45.14|22.27|37.25|
|Ours|44.56|20.93|35.34|

The differences for XSum seems to be [a known issue](https://github.com/pytorch/fairseq/issues/1971). 