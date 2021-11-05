# Hessel et al. (2021)

## Publication
[CLIPScore: A Reference-free Evaluation Metric for Image Captioning](https://arxiv.org/abs/2104.08718)

## Repositories
https://github.com/jmhessel/clipscore

## Available Models
- CLIPScore:
  - Description: A reference-free evaluation metric for image captioning
  - Name: `hessel2021-clipscore`
  - Usage:
    ```python
    from repro.models.hessel2021 import CLIPScore
    model = CLIPScore()
    inputs = [
        {"candidate": "The caption", "image_file": "path/to/image/file.jpeg"}
    ]
    macro, micro = model.predict_batch(inputs)
    # References are optional
    inputs = [
        {"candidate": "The caption", "image_file": "path/to/image/file.jpeg", "references": ["The first", "The second"]}
    ]
    macro, micro = model.predict_batch(inputs)
    ```
    `macro` and `micro` are the average and input-level scores of CLIPScore.
    
## Implementation Notes
**Currently, the CPU and GPU versions give very different scores.**
The CPU score is correct.
I suspect that the GPU version does not work because CLIP uses float16 training, and this error may be specific to the GPU that we tested it on.
Before using the GPU version, you should make sure the unit tests pass (see below).
    
## Docker Information
- Image name: `danieldeutsch/hessel2021:1.0`
- Build command:
  ```shell script
  repro setup hessel2021
  ```
- Requires network: No
  
## Testing
Explain how to run the unittests for this model
```shell script
repro setup hessel2021
pytest models/hessel2021/tests
```

To run the unit tests on the GPU, run:
```shell script
TEST_DEVICES=0 pytest models/hessel2021/tests
```
where `0` is the GPU device ID you want to run the tests on.

## Status
- [ ] Regression unit tests pass  
See note above about GPU.
- [ ] Correctness unit tests pass  
See note above about GPU.
- [ ] Model runs on full test dataset  
Not tested
- [ ] Predictions approximately replicate results reported in the paper  
Not tested
- [ ] Predictions exactly replicate results reported in the paper  
Not tested

## Changelog
