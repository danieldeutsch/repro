DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
set -e
DEVICE=1

repro predict \
  --model-name liu2019-transformerabs \
  --model-kwargs '{"device": '${DEVICE}'}' \
  --dataset-name cnn_dailymail/3.0.0 \
  --split test \
  --output-writer sacrerouge \
  --output-file ${DIR}/output/cnn_dailymail/transformerabs/predictions.jsonl

repro predict \
  --model-name liu2019-bertsumext \
  --model-kwargs '{"device": '${DEVICE}'}' \
  --dataset-name cnn_dailymail/3.0.0 \
  --split test \
  --output-writer sacrerouge \
  --output-file ${DIR}/output/cnn_dailymail/bertsumext/predictions.jsonl

repro predict \
  --model-name liu2019-bertsumextabs \
  --model-kwargs '{"device": '${DEVICE}'}' \
  --dataset-name cnn_dailymail/3.0.0 \
  --split test \
  --output-writer sacrerouge \
  --output-file ${DIR}/output/cnn_dailymail/bertsumextabs/predictions.jsonl

repro predict \
  --model-name liu2019-bertsumextabs \
  --model-kwargs '{"device": '${DEVICE}', "model": "bertsumextabs_xsum.pt"}' \
  --dataset-name xsum/1.2.0 \
  --split test \
  --output-writer sacrerouge \
  --output-file ${DIR}/output/xsum/bertsumextabs/predictions.jsonl

for model in "transformerabs" "bertsumext" "bertsumextabs"; do
  for dataset in "cnn_dailymail" "xsum"; do
    if [ -f ${DIR}/output/${dataset}/${model}/predictions.jsonl ]; then
      python models/sacrerouge/scripts/calculate_rouge.py \
        --input-file ${DIR}/output/${dataset}/${model}/predictions.jsonl \
        --output-file ${DIR}/output/${dataset}/${model}/rouge.json
    fi
  done
done
