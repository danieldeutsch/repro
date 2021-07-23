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
- [ ] Correctness unit tests pass  
- [ ] Model runs on full test dataset  
- [ ] Predictions approximately replicate results reported in the paper  
- [ ] Predictions exactly replicate results reported in the paper  

## Misc
See these [notes](Notes.md) if this code is extended to include training.
