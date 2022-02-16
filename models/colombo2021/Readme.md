# Colombo et al., (2021, 2022); Staerman et al., (2022)

## Publication
This Dockerfile corresponds to three different papers:
- [Automatic Text Evaluation through the Lens of Wasserstein Barycenters](https://arxiv.org/abs/2108.12463)
- [InfoLM: A New Metric to Evaluate Summarization & Data2Text Generation](https://arxiv.org/abs/2112.01589)
- [A Pseudo-Metric between Probability Distributions based on Depth-Trimmed Regions](https://arxiv.org/abs/2103.12711)

## Repositories
All three metrics are implemented in https://github.com/PierreColombo/nlg_eval_via_simi_measures

## Available Models
- BaryScore
  - Name: `colombo2021-baryscore`
  - Usage:
    ```python
    from repro.models.colombo2021 import BaryScore
    model = BaryScore()
    inputs = [
        {"candidate": "The candidate", "references": ["The first reference", "The second"]}
    ]
    macro, micro = model.predict_batch(inputs)
    ```
    `micro` contains the per-input scores and `macro` contains the averaged scores.
    
- InfoLM
  - Name: `colombo2021-infolm`
  - Usage:
    ```python
    from repro.models.colombo2021 import InfoLM
    model = InfoLM()
    inputs = [
        {"candidate": "The candidate", "references": ["The first reference", "The second"]}
    ]
    macro, micro = model.predict_batch(inputs)
    ```
    `micro` contains the per-input scores and `macro` contains the averaged scores.

- DepthScore
  - Name: `colombo2021-depthscore`
  - Usage:
    ```python
    from repro.models.colombo2021 import DepthScore
    model = DepthScore()
    inputs = [
        {"candidate": "The candidate", "references": ["The first reference", "The second"]}
    ]
    macro, micro = model.predict_batch(inputs)
    ```
    `micro` contains the per-input scores and `macro` contains the averaged scores.
    
## Implementation Notes
- For some reason, the unit tests pass on some machines and not on others.
On one of our dev machines, the CPU and GPU tests pass.
On another, the CPU pass but the GPU do not.
On GitHub Actions, the CPU tests do not pass.
Since they are being run in Docker, I assume there is some difference in hardware causing this, but I do not know what the issue is.
    
## Docker Information
- Image name: `danieldeutsch/colombo2021:1.0`
- Build command: Provide documentation on how to build the image
  ```shell script
  repro setup colombo2021
  ```
- Requires network: Yes, it sends a request for resources
  
## Testing
```shell script
repro setup colombo2021
pytest models/colombo2021/tests
```

## Status
- [ ] Regression unit tests pass  
See the implementation notes; https://github.com/danieldeutsch/repro/runs/5210482796
- [ ] Correctness unit tests pass  
- [ ] Model runs on full test dataset  
- [ ] Predictions approximately replicate results reported in the paper  
- [ ] Predictions exactly replicate results reported in the paper  

## Changelog
