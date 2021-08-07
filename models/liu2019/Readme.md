# Liu & Lapata (2019)

## Publication
[Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345)

## Relevant Repositories
https://github.com/nlpyang/PreSumm

## Available Models
The original GitHub repository provides 4 pretrained models:

- [CNN/DM TransformerAbs](https://drive.google.com/file/d/1yLCqT__ilQ3mf5YUUCw9-UToesX5Roxy/view)
  - Description: Their baseline abstractive model trained on the CNN/DailyMail dataset
  - Name: `liu2019-transformerabs`
  - Usage:
    ```python
    from repro.models.liu2019 import TransformerAbs
    model = TransformerAbs()
    summary = model.predict("document")
    ```

- [CNN/DM BertSumExt](https://drive.google.com/file/d/1kKWoV0QCbeIuFt85beQgJ4v0lujaXobJ/view)
  - Description: A BERT-based extractive model trained on the CNN/DailyMail dataset
  - Name: `liu2019-bertsumext`
  - Usage:
    ```python
    from repro.models.liu2019 import BertSumExt
    model = BertSumExt()
    summary = model.predict("document")
    ```

- [CNN/DM BertSumExtAbs](https://drive.google.com/file/d/1-IKVCtc4Q-BdZpjXc4s70_fRsWnjtYLr/view)
  - Description: A BERT-based abstractive model trained on the CNN/DailyMail dataset
  - Name: `liu2019-bertsumextabs`
  - Usage:
    ```python
    from repro.models.liu2019 import BertSumExtAbs
    model = BertSumExtAbs()  # or BertSumExtAbs("bertsumextabs_cnndm.pt")
    summary = model.predict("document")
    ```

- [XSum BertSumExtAbs](https://drive.google.com/file/d/1H50fClyTkNprWJNh10HWdGEdDdQIkzsI/view)
  - Description: A BERT-based abstractive model trained on the XSum dataset
  - Name: `liu2019-bertsumextabs`
  - Usage:
    ```python
    from repro.models.liu2019 import BertSumExtAbs
    model = BertSumExtAbs("bertsumextabs_xsum.pt")
    summary = model.predict("document")
    ```
  
## Implementation Notes
- The input to the pretrained models is expected to be already preprocessed.
Therefore, we tried to replicate their preprocessing steps as closely as we could, which means all of the input documents are tokenized and sentence split using the Stanford CoreNLP library within the docker container.

- If you pass in a pre-sentence tokenized document, the current implementation does not respect those sentence boundaries and will reprocess the document.


## Dockerfile Information
- Image name: `liu2019`
- Build command:
    ```
    repro setup liu2019 \
        [--not-transformerabs-cnndm] \
        [--not-bertsumext-cnndm] \
        [--not-bertsumextabs-cnndm] \
        [--not-bertsumextabs-xsum] \
        [--silent]
    ````
  Each of the flags indicates whether the corresponding model should be not downloaded (all are by default).
- Requires network: No
  
## Testing
```
repro setup liu2019
pytest -s models/liu2019/tests
```

## Status
- [x] Regression unit tests pass  
See the latest successful tests on Github [here](https://github.com/danieldeutsch/repro/actions/runs/1054803359)
- [ ] Correctness unit tests pass  
The authors provide their model outputs and instructions for processing the data from scratch.
We did not attempt to perfectly reproduce their summaries.   
- [x] Model runs on full test dataset  
See [here](experiments/reproduce-results/Readme.md)
- [ ] Predictions approximately replicate results reported in the paper  
The results for the abstractive models approximately replicate the reported in the paper, but the extractive model does not.
See [this experiment](experiments/reproduce-results/Readme.md) for details.
Calculating the ROUGE scores against the original references compared to the references which were preprocessed in the same way as in training did not seem to make a significant difference.

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

  The abstractive models seem to be faithful reproductions of the original results, whereas the extractive model is not.
  It is not clear why.
  
- [ ] Predictions exactly replicate results reported in the paper  
See above
