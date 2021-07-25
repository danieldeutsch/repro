## Description
This experiment runs the oracle sentence-guided model and evaluates its predictions with ROUGE to compare to what is reported in the paper.

## Usage
Running this experiments requires having ROUGE setup (see [here](../../../sacrerouge/Readme.md)).
```shell script
sh models/dou2021/experiments/reproduce-results/run.sh
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