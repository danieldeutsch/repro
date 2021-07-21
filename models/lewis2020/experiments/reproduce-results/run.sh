DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
set -e
DEVICE=1

repro predict \
  --model-name lewis2020-bart \
  --model-args '{"device": '${DEVICE}'}' \
  --dataset-name cnn_dailymail \
  --split test \
  --output-writer sacrerouge \
  --output-file ${DIR}/output/cnn_dailymail/predictions.jsonl

repro predict \
  --model-name lewis2020-bart \
  --model-args '{"device": '${DEVICE}', "pretrained_model": "bart.large.xsum"}' \
  --dataset-name xsum \
  --split test \
  --output-writer sacrerouge \
  --output-file ${DIR}/output/xsum/predictions.jsonl
