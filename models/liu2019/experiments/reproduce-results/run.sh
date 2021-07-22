DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
set -e
DEVICE=1

repro predict \
  --model-name liu2019-transformerabs \
  --model-args '{"device": '${DEVICE}'}' \
  --dataset-name cnn_dailymail/3.0.0 \
  --split test \
  --output-writer sacrerouge \
  --output-file ${DIR}/output/cnn_dailymail/transformerabs/predictions.jsonl

repro predict \
  --model-name liu2019-bertsumext \
  --model-args '{"device": '${DEVICE}'}' \
  --dataset-name cnn_dailymail/3.0.0 \
  --split test \
  --output-writer sacrerouge \
  --output-file ${DIR}/output/cnn_dailymail/bertsumext/predictions.jsonl

repro predict \
  --model-name liu2019-bertsumextabs \
  --model-args '{"device": '${DEVICE}'}' \
  --dataset-name cnn_dailymail/3.0.0 \
  --split test \
  --output-writer sacrerouge \
  --output-file ${DIR}/output/cnn_dailymail/bertsumextabs/predictions.jsonl

repro predict \
  --model-name liu2019-bertsumextabs \
  --model-args '{"device": '${DEVICE}', "model": "bertsumextabs_xsum.pt"}' \
  --dataset-name xsum/1.2.0 \
  --split test \
  --output-writer sacrerouge \
  --output-file ${DIR}/output/xsum/bertsumextabs/predictions.jsonl

python ${DIR}/tokenize_references.py \
  --input-jsonl ${DIR}/output/cnn_dailymail/transformerabs/predictions.jsonl \
  --output-jsonl ${DIR}/output/cnn_dailymail/transformerabs/predictions-tokenized.jsonl

python ${DIR}/tokenize_references.py \
  --input-jsonl ${DIR}/output/cnn_dailymail/bertsumext/predictions.jsonl \
  --output-jsonl ${DIR}/output/cnn_dailymail/bertsumext/predictions-tokenized.jsonl

python ${DIR}/tokenize_references.py \
  --input-jsonl ${DIR}/output/cnn_dailymail/bertsumextabs/predictions.jsonl \
  --output-jsonl ${DIR}/output/cnn_dailymail/bertsumextabs/predictions-tokenized.jsonl

python ${DIR}/tokenize_references.py \
  --input-jsonl ${DIR}/output/xsum/bertsumextabs/predictions.jsonl \
  --output-jsonl ${DIR}/output/xsum/bertsumextabs/predictions-tokenized.jsonl
