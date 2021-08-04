cd factCC/modeling
wandb disabled
echo '{"id": "1", "claim": "this is the claim", "text": "This is the text", "label": "CORRECT"}' > data-dev.jsonl
python run_test.py \
  --task_name factcc_annotated \
  --do_test \
  --eval_all_checkpoints \
  --do_lower_case \
  --max_seq_length 512 \
  --per_gpu_train_batch_size 12 \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --data_dir . \
  --output_dir ../../factcc-checkpoint \
  --output-file output.jsonl
rm data-dev.jsonl output.jsonl