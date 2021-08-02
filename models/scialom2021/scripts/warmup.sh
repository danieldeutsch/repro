echo '{"candidate": "Dan went to the store.", "source": "Dan went to the grocery store.", "reference": "Dan went to buy food."}' > input.jsonl
python score.py \
  --input-file input.jsonl \
  --kwargs '{"task": "summarization", "do_weighter": true}' \
  --cuda-device -1 \
  --output-file output.jsonl
rm input.jsonl output.jsonl