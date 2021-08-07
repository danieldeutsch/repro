# Lewis et al. (2020)

## Publication
[BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)

## Relevant Repositories
https://github.com/pytorch/fairseq/tree/master/examples/bart

## Available Models
The original GitHub repository provides 2 pretrained models:

- [bart.large.cnn](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz)
  - Description: A model trained on the CNN/DailyMail dataset
  - Name: `lewis2020-bart`
  - Usage:
    ```python
    from repro.models.lewis2020 import BART
    model = BART()  # or BART("bart.large.cnn")
    summary = model.predict("document")
    ```

- [bart.large.xsum](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.xsum.tar.gz)
  - Description: A model trained on the XSum dataset
  - Name: `lewis2020-bart`
  - Usage:
    ```python
    from repro.models.lewis2020 import BART
    model = BART("bart.large.xsum")
    summary = model.predict("document")
    ```
  
## Implementation Notes
This implementation is based on the original code in the `fairseq` library, not the `transformers` library.

## Dockerfile Information
- Image name: `lewis2020`
- Build command:
    ```
    repro setup lewis2020 \
        [--not-cnndm] \
        [--not-xsum] \
        [--silent]
    ````
  Each of the flags indicates whether the model trained on the corresponding dataset should not be downloaded (both are by default).
- Requires network: Yes.
Even with running a warmup query, [the inference sends a request to retrieve the etag of a file](https://github.com/pytorch/fairseq/blob/72323586aeae75e2b704c1c936784471bfa75019/fairseq/file_utils.py#L278), which fails if the network is disabled.

## Testing
```
repro setup lewis2020
pytest models/lewis2020/tests
```

## Status
- [x] Regression unit tests pass  
See latest run [here](https://github.com/danieldeutsch/repro/actions/runs/1054130606).
- [ ] Correctness unit tests pass  
None are provided in the original repo
- [x] Model runs on full test dataset  
Successfully run on both CNN/DailyMail and XSum (see [here](experiments/reproduce-results/Readme.md))
- [ ] Predictions approximately replicate results reported in the paper  
We reran the models on the CNN/DailyMail and XSum datasets (see [here](experiments/reproduce-results/Readme.md)) and calculated the ROUGE score.
The CNN/DailyMail results are very close to those in the paper:

  ||R1|R2|RL|
  |-|-|-|-|
  |Reported|44.16|21.28|40.90|
  |Ours|44.31|21.12|41.18|

  This seems to be a faithful implementation for `bart.large.cnn`.

  For XSum, the results are not as close

  ||R1|R2|RL|
  |-|-|-|-|
  |Reported|45.14|22.27|37.25|
  |Ours|44.56|20.93|35.34|

  However, this seems to be [a known issue](https://github.com/pytorch/fairseq/issues/1971).

- [ ] Predictions exactly replicate results reported in the paper  
No, see above.
