cd SUPERT
mkdir -p temp/0/input_docs temp/0/summaries
printf "<TEXT>\nThe first input document\n</TEXT>" > temp/0/input_docs/0.txt
printf "<TEXT>\nThe second input document\n</TEXT>" > temp/0/input_docs/1.txt
printf "The summary to score" > temp/0/summaries/0
python run_batch.py temp output.jsonl
rm -r temp output.jsonl
