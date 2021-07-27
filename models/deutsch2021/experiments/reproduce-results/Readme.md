## Description
This experiment evaluates the pre-trained QA model on the SQuAD v2 dataset.

## Usage
```shell script
sh models/deutsch2021/experiments/reproduce-results/run.sh
``` 

## Results
The `metrics.json` output file contains:
```json
{
  "exact": 43.44310620736124,
  "f1": 46.7579012131542,
  "total": 11873,
  "HasAns_exact": 87.01079622132254,
  "HasAns_f1": 93.64989222398445,
  "HasAns_total": 5928,
  "NoAns_exact": 0.0,
  "NoAns_f1": 0.0,
  "NoAns_total": 5945,
  "best_exact": 86.86936747241641,
  "best_exact_thresh": 0.003306884248849975,
  "best_f1": 89.78113205776883,
  "best_f1_thresh": 0.003306884248849975
}
```
These are the expected numbers based on [here](https://github.com/CogComp/qaeval-experiments/tree/master/models/answering).
Therefore, we conclude this implementation reproduces the original model's results.
