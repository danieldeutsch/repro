# SQuAD v2

## Publication
[Know What You Don't Know: Unanswerable Questions for SQuAD](https://arxiv.org/abs/1806.03822)

## Repositories
https://worksheets.codalab.org/worksheets/0xbe2859a20b9e41d2a2b63ea11bd97740

## Available Models
This implementation wraps the evaluation for SQuAD v2 on the official dev set.
The dev set is equivalent to predicting on the "validation" split using the `HuggingfaceDatasetsDatasetReader` with the `dataset_name="squad_v2"`. 

- Evaluation
  - Description: The SQuAD-v2 evaluation script for the dev set.
  - Name: `squad-v2`
  - Usage:
    ```python
    from repro.models.squad_v2 import SQuADv2Evaluation
    model = SQuADv2Evaluation()
    inputs = [
        {"instance_id": "56ddde6b9a695914005b9628", "prediction": "France", "null_probability": 4.3727909708190646e-07}
    ]
    metrics = model.predict_batch(inputs)
    ```
    
## Implementation Notes
    
## Docker Information
- Image name: `squad-v2`
- Build command:
  ```shell script
  repro setup squad-v2 [--silent]
  ```
- Requires network: No
  
## Testing
n/a

## Status
Appears to work as expected.  
