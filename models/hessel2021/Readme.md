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
Running the metric on CPU versus GPU may give slightly different results.
See the [original code's Readme](https://github.com/jmhessel/clipscore/blob/main/README.md#reproducibility-notes) for more info.
    
## Docker Information
- Image name: `danieldeutsch/hessel2021:1.0`
- Build command:
  ```shell script
  repro setup hessel2021
  ```
- Requires network: No
  
## Testing
```shell script
repro setup hessel2021
pytest models/hessel2021/tests
```

## Status
- [x] Regression unit tests pass  
- [x] Correctness unit tests pass  
See [here](https://github.com/danieldeutsch/repro/actions/runs/1426809122).
- [ ] Model runs on full test dataset  
Not tested
- [ ] Predictions approximately replicate results reported in the paper  
Not tested
- [ ] Predictions exactly replicate results reported in the paper  
Not tested

## Changelog
