# Dugan et al. (2020)

## Publication
[RoFT: A Tool for Evaluating Human Detection of Machine-Generated Text](https://arxiv.org/abs/2010.03070)

## Repositories
https://github.com/kirubarajan/roft

## Available Models
- GPT2-XL Recipe Generation Model
  - Description: A GPT2-XL model fine-tuned on Recipe1M+ dataset, available at `gs://roft_saved_models/gpt2-xl_recipes.tar.gz`
  - Name: `dugan2020-roft-recipe`
  - Usage:
    ```python
    from repro.models.dugan2020 import RoFTRecipeGenerator
    name = "Redwood Room Apple Pie"
    ingredients = [
      "1 tablespoon cornstarch",
      "12 cup sugar",
      "14 cup cream",
      "1 tablespoon lemon juice",
      "3 tablespoons butter",
      "20 ounces apples, slices",
      "9 inches pie crusts, baked",
      "8 ounces cream cheese",
      "13 cup sugar",
      "1 egg",
      "12 cup coconut",
      "12 cup walnuts, chopped"
    ]
    model = RoFTRecipeGenerator()
    recipe = model.predict(name, ingredients)
    ```
    
## Implementation Notes
- Although you can set the random seed, the results on the CPU versus the GPU may be different.
    
## Docker Information
- Image name: `dugan2020`
- Build command:
  ```shell script
  repro setup dugan2020 [--silent]
  ```
- Requires network: No
  
## Testing
Explain how to run the unittests for this model
```
repro setup dugan2020 
pytest models/dugan2020/tests
```

## Status
- [x] Regression unit tests pass  
The model takes up too much memory to run on Github. See [here](https://github.com/danieldeutsch/repro/actions/runs/1057564558)
- [ ] Correctness unit tests pass  
No expected outputs provided by the original code
- [ ] Model runs on full test dataset  
Not tested
- [ ] Predictions approximately replicate results reported in the paper  
Not tested
- [ ] Predictions exactly replicate results reported in the paper  
Not tested


