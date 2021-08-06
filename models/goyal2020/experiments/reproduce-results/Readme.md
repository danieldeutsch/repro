## Description
This experiment reproduces the results from Section 6.1 of the paper, which calculates the accuracy of reranking 373 correct/incorrect sentences from summarization data collected by Falke et al. (2019).

## Usage
Run the commands from this experiment directory.
```shell script
# Download the input data
wget https://raw.githubusercontent.com/tagoyal/dae-factuality/main/resources/summary_pairs.json

python run.py \
    --input-file summary_pairs.json \
    --device 1 \
    --output-file output.json
```

## Results
The output from our script is:
```json
{
  "dae_basic": 0.8337801608579088,
  "dae_w_syn": 0.8364611260053619,
  "dae_w_syn_hallu": 0.7801608579088471
}
```
The paper reports 83.6% accuracy for "dae_w_syn", so this experiment reproduces that result.
The other models' scores are not reported.