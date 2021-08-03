mkdir -p models

echo '{"candidate": "This is the summary", "references": ["This is the reference summary"]}' > input.jsonl
python score.py \
  --input-file input.jsonl \
  --kwargs '{"cuda_device": -1, "use_lerc": true}' \
  --output-file output.jsonl
rm input.jsonl output.jsonl
