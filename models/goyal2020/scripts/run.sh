set -e

input=$1
output=$2
model=$3
device=$4
sleep=$5

# Start the CoreNLP server and sleep
cd stanford-corenlp-full-2018-01-31
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer &
CORENLP_PID=$!
sleep ${sleep}

# Run the scoring script
cd ../dae-factuality
python score.py \
  --input-file ${input} \
  --model_type electra_dae \
  --input_dir ${model} \
  --gpu_device ${device} \
  --output-file ${output}

# Terminate the CoreNLP server
kill ${CORENLP_PID}