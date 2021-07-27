# Dou et al. (2021)

## Publication
[GSum: A General Framework for Guided Neural Abstractive Summarization](https://arxiv.org/abs/2010.08014)

## Repositories
https://github.com/neulab/guided_summarization

## Available Models
The authors have released their BART-based model that uses sentence guidance, downloadable [here](https://drive.google.com/file/d/1BMKhAh2tG5p8THxugZWMPc7NXqwJDHLw/view?usp=sharing).

- Oracle Sentence-Guided
  - Description: A BART-based model trained with sentence supervision.
  This model will use the oracle supervision, which is calculated by selecting document sentences which maximize the ROUGE score with respect to the ground-truth.
  Therefore, it requires the reference summaries as input.
  - Name: `dou2021-oracle-sentence-gsum`
  - Usage:
    ```python
    from repro.models.dou2021 import OracleSentenceGSumModel
    model = OracleSentenceGSumModel()
    summary = model.predict("document", "reference")
    ```
    
- Sentence-Guided
  - Description: A BART-based model trained with sentence supervision.
  This is the same pre-trained model as the oracle sentence-guided supervision, but it instead uses [`BertSumExt`](../liu2019/Readme.md) to extract the sentence guidance.
  Running this model also requires having the `BertSumExt` model setup with the model pre-trained on CNN/DailyMail.
  We caution that the [`BertSumExt`](../liu2019/Readme.md) currently does not reproduce the results in the respective paper.
  - Name: `dou2021-sentence-gsum`
  - Usage:
    ```python
    from repro.models.dou2021 import SentenceGSumModel
    model = SentenceGSumModel()
    summary = model.predict("document")
    ```
    
## Implementation Notes
    
## Docker Information
- Image name: `dou2021`
- Build command:
  ```shell script
  repro setup dou2021 [--silent]
  ```
- Requires network: Yes.
`fairseq` sends a request to retrieve an etag even if the file is present.
This also happens with [Lewis et al. (2020)](../lewis2020/Readme.md).
  
## Testing
```shell script
repro setup dou2021
pytest models/dou2021/tests
```

## Status
- [x] Regression unit tests pass  
Only the tests which require models from Liu & Lapata (2019) fail on Github.
See [here](https://github.com/danieldeutsch/repro/actions/runs/1071401161).
- [ ] Correctness unit tests pass  
- [x] Model runs on full test dataset  
See [here](experiments/reproduce-results/Readme.md)
- [x] Predictions approximately replicate results reported in the paper  
The MatchSum-guided BART-based model was the only BART-based result reported in the paper, and it does come sufficiently close.

  ||R1|R2|RL|
  |-|-|-|-|
  |Reported|45.94|22.32|42.48|
  |Ours|45.80|22.18|42.44|

  See [here](experiments/reproduce-results/Readme.md) for more details about the different model variants.
  
- [ ] Predictions exactly replicate results reported in the paper

## Misc
See these [notes](Notes.md) if this code is extended to include training.
