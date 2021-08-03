cd feqa
echo '{"candidate": "This is the candidate summary.", "source": "This is the source document."}' > input.jsonl
python score.py \
  --input-file input.jsonl \
  --cuda-device -1 \
  --batch-size 2 \
  --output-file output.jsonl
rm input.jsonl output.jsonl