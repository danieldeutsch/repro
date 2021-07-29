mkdir -p models

echo '{"instance_id": "1", "summarizer_id": "warmup", "summarizer_type": "peer", "summary": {"text": "This is the summary"}, "references": [{"text": "This is the reference summary"}]}' > input.jsonl
sacrerouge qa-eval evaluate \
  --input-files input.jsonl \
  --dataset-reader reference-based \
  --use_lerc true \
  --generation_batch_size 1 \
  --answering_batch_size 1 \
  --lerc_batch_size 1 \
  --cuda_device -1 \
  --macro-output-json macro.json \
  --micro-output-jsonl micro.jsonl
rm input.jsonl macro.json micro.jsonl
