echo '{"candidate": "First translation", "references": ["First reference", "The second"]}' > input.jsonl
for model_name in ${MODELS}; do
  python score.py \
    --input-file input.jsonl \
    --model-name ${model_name} \
    --cuda-device -1 \
    --output-file output.txt
done
rm input.jsonl output.txt