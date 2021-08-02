echo '{"candidate": "The candidate", "references": ["The first", "The second reference"]}' > input.jsonl
python score.py \
  --input-file input.jsonl \
  --use-stopwords true \
  --output-file output.jsonl
rm input.jsonl output.jsonl