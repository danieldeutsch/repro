cd nrl-qasrl
echo '{"sentence": "This is an example sentence"}' > input.jsonl
python -m allennlp.run predict \
  ./data/qasrl_parser_elmo input.jsonl \
  --include-package nrl \
  --predictor qasrl_parser \
  --output-file output.jsonl
rm input.jsonl output.jsonl