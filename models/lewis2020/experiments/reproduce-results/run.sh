DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
set -e
DEVICE=1

repro predict \
  --model-name lewis2020-bart \
  --model-kwargs '{"device": '${DEVICE}'}' \
  --dataset-name cnn_dailymail/3.0.0 \
  --split test \
  --output-writer sacrerouge \
  --output ${DIR}/output/cnn_dailymail/predictions.jsonl

repro predict \
  --model-name lewis2020-bart \
  --model-kwargs '{"device": '${DEVICE}', "model": "bart.large.xsum"}' \
  --dataset-name xsum/1.2.0 \
  --split test \
  --output-writer sacrerouge \
  --output ${DIR}/output/xsum/predictions.jsonl

for dataset in "cnn_dailymail" "xsum"; do
  if [ -f ${DIR}/output/${dataset}/predictions.jsonl ]; then
    repro predict \
      --model-name sacrerouge-rouge \
      --input-files ${DIR}/output/${dataset}/predictions.jsonl \
      --dataset-reader sacrerouge \
      --output-writer metrics \
      --output ${DIR}/output/${dataset}/rouge.json
  fi
done
