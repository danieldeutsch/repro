cd BARTScore
echo '{"candidate": "The candidate text", "references": ["The reference text"]}' > input.jsonl
for model in default cnn; do
  python score.py \
    --input-file input.jsonl \
    --device -1 \
    --batch-size 1 \
    --model ${model} \
    --output-file output.jsonl
done
rm input.jsonl
rm output.jsonl