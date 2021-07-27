mkdir -p models

if [ "$QG" = "true" ]; then
  echo '{"context": "My name is Dan.", "start": 11, "end": 14}' > input.jsonl
  python generate_questions.py \
    --input-file input.jsonl \
    --model-file models/question-generation.model.tar.gz \
    --cuda-device -1 \
    --batch-size 2 \
    --output-file output.jsonl
  rm input.jsonl output.jsonl
fi

if [ "$QA" = "true" ]; then
  echo '{"context": "My name is Dan.", "question": "What is my name?"}' > input.jsonl
  python answer_questions.py \
    --input-file input.jsonl \
    --model-dir models/question-answering \
    --cuda-device -1 \
    --batch-size 2 \
    --output-file output.jsonl
  rm input.jsonl output.jsonl
fi