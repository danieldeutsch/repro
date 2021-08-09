echo '{"candidate": "This is the candidate", "source": "This is the source article."}' > input.jsonl
python score.py \
  --input-file input.jsonl \
  --output-file output.jsonl
rm input.jsonl output.jsonl