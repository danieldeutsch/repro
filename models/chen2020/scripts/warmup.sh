cd MOCHA
echo '{}' > input.jsonl
python predict.py \
  --input-file input.jsonl \
  --model-path ../lerc-2020-11-18.tar.gz \
  --batch-size 1 \
  --cuda-device -1 \
  --output-file output.jsonl
rm input.jsonl output.jsonl
