DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
set -e
DEVICE=1

repro predict \
  --model-name deutsch2021-question-answering \
  --model-kwargs '{"device": '${DEVICE}'}' \
  --dataset-name squad_v2 \
  --split validation \
  --predict-kwargs '{"return_dicts": true}' \
  --output ${DIR}/output/question-answering/predictions.jsonl

repro predict \
  --model-name squad-v2 \
  --input-files ${DIR}/output/question-answering/predictions.jsonl \
  --dataset-reader deutsch2021-question-answering-eval \
  --output-writer metrics \
  --output ${DIR}/output/question-answering/metrics.json
