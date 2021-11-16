echo '{"document": "The input document", "summaries": ["The summary"]}' > input.jsonl
for type in help tune; do
  python score.py \
    --input-file input.jsonl \
    --type ${type} \
    --device -1 \
    --random-seed 123 \
    --kwargs '{}' \
    --output-file output.json
done
rm input.jsonl
rm output.json