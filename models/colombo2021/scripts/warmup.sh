cd nlg_eval_via_simi_measures

echo "I like my cakes very much" > ref.txt
echo "I like my cakes very much" > hyp.txt

for metric in infolm baryscore depthscore; do
  python score_cli.py \
    --ref ref.txt \
    --cand hyp.txt \
    --metric_name=${metric} \
    --output_file out.txt
done

rm ref.txt hyp.txt out.txt