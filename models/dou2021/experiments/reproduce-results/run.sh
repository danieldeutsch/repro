DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
set -e
DEVICE=1

if [[ ! -d ${DIR}/output/cnn_dm ]]; then
  gdown https://drive.google.com/uc?id=1goxgX-0_2Jo7cNAFrsb9BJTsrH7re6by
  tar -xvf cnn_dm.tar.gz
  rm cnn_dm.tar.gz
  mv cnn_dm ${DIR}/output
fi

if [[ ! -d ${DIR}/output/oracle ]]; then
  gdown https://drive.google.com/uc?id=12SpWwfD3syIxcC-SdSNnDOI5sbXJaylC
  unzip bart_sentence_guide_cnndm.zip
  rm bart_sentence_guide_cnndm.zip
  mkdir -p ${DIR}/output/oracle
  mv train.oracle val.oracle test.oracle test.matchsum ${DIR}/output/oracle
fi

for model in "dou2021-oracle-sentence-gsum"; do
#  repro predict \
#    --model-name ${model} \
#    --model-args '{"device": '${DEVICE}'}' \
#    --dataset-name cnn_dailymail/3.0.0 \
#    --split test \
#    --output-writer sacrerouge \
#    --output-file ${DIR}/output/cnn_dailymail/${model}/predictions.jsonl
#
  repro predict \
    --model-name ${model} \
    --model-args '{"device": '${DEVICE}'}' \
    --dataset-reader dou2021 \
    --input-files ${DIR}/output/cnn_dm/test.source ${DIR}/output/cnn_dm/test.target ${DIR}/output/oracle/test.oracle \
    --output-writer sacrerouge \
    --output-file ${DIR}/output/cnn_dailymail-preproc/${model}/predictions.jsonl
done
