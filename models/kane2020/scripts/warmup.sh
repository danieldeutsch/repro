echo '{"candidate": "The dinner did not taste good.", "reference": "The dinner was delicious."}' > input.jsonl
python score.py \
  --input-file input.jsonl \
  --six-dim false \
  --aggregator agg_two \
  --output-file output.jsonl
rm input.jsonl output.jsonl