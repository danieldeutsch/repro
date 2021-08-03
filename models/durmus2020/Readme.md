# Durmus et al. (2020)

## Publication
[FEQA: A Question Answering Evaluation Framework for Faithfulness Assessment in Abstractive Summarization](https://aclanthology.org/2020.acl-main.454/)

## Repositories
https://github.com/esdurmus/feqa

## Available Models
- FEQA
  - Description: A document-based faithfulness evaluation metric
  - Name: `durmus2020-feqa`
  - Usage:
    ```python
    from repro.models.durmus2020 import FEQA
    model = FEQA()
    inputs = [
        {"candidate": "The candidate text", "sources": ["The source"]}
    ]
    macro, micro = model.predict_batch(inputs)
    ```
    The output `macro` is the FEQA score averaged over the inputs.
    `micro` is the invidiaul FEQA scores per input. 
    
## Implementation Notes
- FEQA only supports single source documents, so the length of `sources` must be 1

- The [example](https://github.com/esdurmus/feqa/blob/master/run_feqa.ipynb) shows that `en_core_web_sm==2.1.0`.
  However, when we ran the code with that version, we encountered the following error:
  
  > ValueError: [E167] Unknown morphological feature: 'ConjType' (9141427322507498425).
  This can happen if the tagger was trained with a different set of morphological features.
  If you're using a pretrained model, make sure that your models are up to date

  We instead use `en_core_web_sm==2.3.1`, which did not cause this error.
  
- We are unable to test this implementation on a GPU because the process consumes more than 12gb of memory.
  The error comes during the question-answering step when the SQuAD model is called.
    
## Docker Information
- Image name: `durmus2020`
- Build command:
  ```shell script
  repro setup durmus2020 [--silent]
  ```
- Requires network: No
  
## Testing
```shell script
repro setup durmus2020 
pytest models/durmus2020/tests
```

## Status
- [x] Regression unit tests pass  
We were unable to run tests on the MultiLing2011 data that we have used to regression testing other metrics because the process runs out of memory on the GPU (ours is 12gb) and the CPU version is too slow.
However, we do run a test on some toy examples. 
- [x] Correctness unit tests pass  
See [here](https://github.com/danieldeutsch/repro/actions/runs/1095392445).
The [example](https://github.com/esdurmus/feqa/blob/master/run_feqa.ipynb) input and output ran successfully when we used `en_core_web_sm==2.1.0`, but after we changed to `en_core_web_sm==2.3.1`, one of the examples changed its score slightly.
However, it is close enough that we consider it to be ok especially since it was correct with the original Spacy model.
- [ ] Model runs on full test dataset  
Not tested
- [ ] Predictions approximately replicate results reported in the paper  
Not tested
- [ ] Predictions exactly replicate results reported in the paper  
Not tested