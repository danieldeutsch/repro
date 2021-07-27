## Description
This experiment runs the oracle sentence-guided model and evaluates its predictions with ROUGE to compare to what is reported in the paper.

## Usage
Running this experiments requires having ROUGE setup (see [here](../../../sacrerouge/Readme.md)).
```shell script
sh models/dou2021/experiments/reproduce-results/run.sh
```

## Results
The paper uses BERT- and BART-based versions of their model.
We report numbers using only the BART models, and the only BART numbers reported in the paper are BART + guidance from MatchSum.
We report ROUGE scores for several variants:

1. Oracle-guided model using the `datasets` version of the CNN/DailyMail dataset, recomputing the oracle sentences.
2. Oracle-guided model using the paper's version of the CNN/DailyMail dataset with their released oracle sentences.
3. BertSumExt-guided model using the `datasets` version of the CNN/DailyMail dataset and using [`BertSumExt`](../../../liu2019/Readme.md) to compute the sentence guidance
4. BertSumExt-guided model using the paper's version of the CNN/DailyMail dataset and using [`BertSumExt`](../../../liu2019/Readme.md) to compute the sentence guidance
5. MatchSum-guided model using the paper's version of the dataset and their released MatchSum sentence guidance.

Oracle Sentence-Guided (`datasets`)
||R1|R2|RL|
|-|-|-|-|
|Reported|-|-|-|
|Ours|52.32|29.18|49.05|

Overall, these are lower than the BERT-based model, which is surprising.

Oracle Sentence-Guided (paper dataset version)
||R1|R2|RL|
|-|-|-|-|
|Reported|-|-|-|
|Ours|52.41|29.31|49.1|

These are roughly equal to the `datasets` version.

BertSumExt Sentence-Guided (`datasets`)
||R1|R2|RL|
|-|-|-|-|
|Reported|-|-|-|
|Ours|44.33|21.03|41.03|

BertSumExt Sentence-Guided (paper dataset version)
||R1|R2|RL|
|-|-|-|-|
|Reported|-|-|-|
|Ours|44.45|21.10|41.09|

These are roughly equal to the `datasets` version.

MatchSum Sentence-Guided (paper dataset version)
||R1|R2|RL|
|-|-|-|-|
|Reported|45.94|22.32|42.48|
|Ours|45.80|22.18|42.44|

Overall, these are pretty close to those reported in the paper.

## Conclusions
- The differences between the `datasets` version of CNN/DailyMail and the paper's seems to not make a difference.
- The BERT-oracle model they report in their paper is higher than our BART-oracle, which is surprising.
- The BertSumExt BART-model gets higher scores than the equivalent BERT-model in the paper.
- The MatchSum-guided model gets comparable results to those in the paper.
