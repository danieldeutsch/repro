DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
set -e
DEVICE=1

repro predict \
  --model-name chen2020-lerc \
  --model-kwargs '{"device": '${DEVICE}'}' \
  --dataset-name mocha \
  --split validation \
  --output-writer-kwargs '{"include_input": true}' \
  --output ${DIR}/output/scores.jsonl

repro predict \
  --model-name chen2020-eval \
  --dataset-reader chen2020-eval \
  --input-files ${DIR}/output/scores.jsonl \
  --output-writer metrics \
  --output ${DIR}/output/metrics.json
