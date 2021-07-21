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
        [--cnndm] \
        [--xsum] \
        [--silent]
    ````
  Each of the flags indicates whether the model trained on the corresponding dataset should be downloaded.
- Requires network: Yes.
Even with running a warmup query, [the inference sends a request to retrieve the etag of a file](https://github.com/pytorch/fairseq/blob/72323586aeae75e2b704c1c936784471bfa75019/fairseq/file_utils.py#L278), which fails if the network is disabled.

## Testing
```
repro setup lewis2020 --cnndm --xsum
pytest models/lewis2020/tests
```

## Status
- [x] Regression unit tests pass  
- [ ] Correctness unit tests pass  
None are provided in the original repo
- [ ] Model runs on full test dataset  
Not tested
- [ ] Predictions approximately replicate results reported in the paper  
Not tested
- [ ] Predictions exactly replicate results reported in the paper  
Not tested
