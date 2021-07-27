## Description
This experiment scores the MOCHA dev set with the pre-trained LERC model and compares the Pearon correlations to those reported in the paper.
The model is trained on all of the datsets (not held-one-out).

## Usage
```shell script
sh models/chen2020/experiments/reproduce-results/run.sh
```

## Results
The MOCHA results are:
```json
{
  "cosmosqa": {
    "overall": 0.8598405422484737,
    "gpt2": 0.6288519634921272,
    "backtranslation": 0.7384899877925727
  },
  "drop": {
    "overall": 0.8164604441733485,
    "naqanet": 0.8147694036910175,
    "nabert": 0.8244779435365354
  },
  "mcscript": {
    "overall": 0.8123871946456581,
    "gpt2": 0.8079166629787261,
    "backtranslation": 0.6220700998984932,
    "mhpg": 0.8063469302202855
  },
  "narrativeqa": {
    "overall": 0.7938505901231165,
    "gpt2": 0.8021701542983071,
    "backtranslation": 0.748864660210755,
    "narrativeqa": 0.5377927669442789,
    "mhpg": 0.8425706566636226
  },
  "quoref": {
    "overall": 0.776692247523106,
    "bert": 0.776692247523106
  },
  "socialiqa": {
    "overall": 0.8146354907194588,
    "gpt2": 0.6628362353743031,
    "backtranslation": 0.6897312959422057
  }
}
```

In comparison to those values reported in the paper, they are close:

|Dataset|Reported|Ours|
|-|-|-|
|NarrativeQA|.805|.794|
|MCScript|.816|.812|
|CosmosQA|.864|.860|
|SocialIQA|.820|.815|
|DROP|.796|.816|
|Quoref|.794|.777|

Overall, the results are very close to the originals.
