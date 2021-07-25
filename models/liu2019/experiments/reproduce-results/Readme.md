## Description
This experiment runs prediction using all of the models on the CNN/DailyMail or XSum dataset and evaluates the predictions using ROUGE to compare to the results reported in the original paper.

## Usage
Running this evaluation also requires having the ROUGE metric setup (see [here](../../../sacrerouge/Readme.md)).
```shell script
sh models/liu2019/experiments/reproduce-results/run.sh
```

## Results
`TransformerAbs` on CNN/DailyMail
||R1|R2|RL|
|-|-|-|-|
|Reported|40.21|17.76|37.09|
|Ours|40.32|17.73|37.18|

`BertSumExt` on CNN/DailyMail
||R1|R2|RL|
|-|-|-|-|
|Reported|43.23|20.24|39.63|
|Ours|41.88|18.89|38.17|

`BertSumExtAbs` on CNN/DailyMail
||R1|R2|RL|
|-|-|-|-|
|Reported|42.13|19.60|39.18|
|Ours|42.02|19.34|39.01|

`BertSumExtAbs` on XSum
||R1|R2|RL|
|-|-|-|-|
|Reported|38.81|16.50|31.27|
|Ours|38.87|16.40|31.30|
