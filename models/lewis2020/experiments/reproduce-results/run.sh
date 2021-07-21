DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
set -e

#repro predict \
#  --model-name lewis2020-bart \
#  --dataset-name cnn_dailymail \
#  --split test \
#  --output-writer sacrerouge \
#  --output-file ${DIR}/output/cnn_dailymail/predictions.jsonl

repro predict \
  --model-name lewis2020-bart \
  --model-args '{"pretrained_model": "bart.large.xsum"}' \
  --dataset-name xsum \
  --split test \
  --output-writer sacrerouge \
  --output-file ${DIR}/output/xsum/predictions.jsonl