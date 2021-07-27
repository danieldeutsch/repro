DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
set -e
DEVICE=1

if [[ ! -d ${DIR}/output/data/input ]]; then
  gdown https://drive.google.com/uc?id=1goxgX-0_2Jo7cNAFrsb9BJTsrH7re6by
  tar -xvf cnn_dm.tar.gz
  rm cnn_dm.tar.gz
  mkdir -p ${DIR}/output/data
  mv cnn_dm ${DIR}/output/data/input
fi

if [[ ! -d ${DIR}/output/data/oracle ]]; then
  gdown https://drive.google.com/uc?id=12SpWwfD3syIxcC-SdSNnDOI5sbXJaylC
  unzip bart_sentence_guide_cnndm.zip
  rm bart_sentence_guide_cnndm.zip
  mkdir -p ${DIR}/output/data/oracle
  mv train.oracle val.oracle test.oracle test.matchsum ${DIR}/output/data/oracle
fi

for model in "dou2021-oracle-sentence-gsum" "dou2021-sentence-gsum"; do
  # Run on the standard version of the dataset
  repro predict \
    --model-name ${model} \
    --model-kwargs '{"device": '${DEVICE}'}' \
    --dataset-name cnn_dailymail/3.0.0 \
    --split test \
    --output-writer sacrerouge \
    --output ${DIR}/output/original/${model}/predictions.jsonl

  # Run on their preprocessed version
  repro predict \
    --model-name ${model} \
    --model-kwargs '{"device": '${DEVICE}'}' \
    --dataset-reader dou2021 \
    --input-files ${DIR}/output/data/input/test.source ${DIR}/output/data/input/test.target \
    --output-writer sacrerouge \
    --output ${DIR}/output/preprocessed/${model}/predictions.jsonl

  if [ "${model}" = "dou2021-oracle-sentence-gsum" ]; then
    # Run with their oracle guidance
    repro predict \
      --model-name ${model} \
      --model-kwargs '{"device": '${DEVICE}'}' \
      --dataset-reader dou2021 \
      --input-files ${DIR}/output/data/input/test.source ${DIR}/output/data/input/test.target ${DIR}/output/data/oracle/test.oracle \
      --output-writer sacrerouge \
      --output ${DIR}/output/oracle/${model}/predictions.jsonl

    # Run with their MatchSum guidance
    repro predict \
      --model-name ${model} \
      --model-kwargs '{"device": '${DEVICE}'}' \
      --dataset-reader dou2021 \
      --input-files ${DIR}/output/data/input/test.source ${DIR}/output/data/input/test.target ${DIR}/output/data/oracle/test.matchsum \
      --output-writer sacrerouge \
      --output ${DIR}/output/matchsum/${model}/predictions.jsonl
  fi

  # Evaluate with ROUGE
  for version in "original" "preprocessed" "oracle" "matchsum"; do
    if [ -f ${DIR}/output/${version}/${model}/predictions.jsonl ]; then
      repro predict \
        --model-name sacrerouge-rouge \
        --input-files ${DIR}/output/${version}/${model}/predictions.jsonl \
        --dataset-reader sacrerouge \
        --output-writer metrics \
        --output ${DIR}/output/${version}/${model}/rouge.json
    fi
  done
done
