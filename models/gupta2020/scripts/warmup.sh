cd nmn-drop
echo '{"passage": "Dummy passage", "question": "Dummy question?"}' > example.jsonl
allennlp predict \
  --output-file output.jsonl \
  --predictor drop_demo_predictor \
  --include-package semqa \
  --silent \
  --batch-size 1 \
  --cuda-device -1 \
  ../model.tar.gz \
  example.jsonl
rm example.jsonl
rm output.jsonl
