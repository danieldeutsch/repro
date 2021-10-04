cd BARTScore
echo '{"candidate": "The candidate text", "references": ["The reference text"]}' > input.jsonl
python score.py \
  --input-file input.jsonl \
  --device -1 \
  --batch-size 1 \
  --use-parabank false \
  --output-file output.jsonl
rm input.jsonl
rm output.jsonl