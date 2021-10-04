cd Lite2-3Pyramid
echo "This is the reference summary" > references.txt
echo "This is the candidate summary" > summaries.txt
echo "1" > ids.txt
mkdir output
python score.py \
  --extract_stus \
  --reference references.txt \
  --doc_id ids.txt \
  --output_dir output \
  --use_coref

# Score the summary using each of the models which
# will pre-cache them
for model_name in ${MODELS}; do
  python score.py \
    --unit output/STUs.txt \
    --summary summaries.txt \
    --model ${model_name} \
    --output_file output/scores.txt
done

rm references.txt
rm ids.txt
rm -r output