input=$1
target=$2
batch_size=$3
output=$4

spm_encode --model=m39v1/spm.model --output_format=piece < ${input} > data.sp.src

awk '{print "<'$target'>"}' data.sp.src > data.sp.tgt

fairseq-preprocess --source-lang src --target-lang tgt \
    --tgtdict m39v1/dict.tgt.txt --srcdict m39v1/dict.src.txt \
    --testpref data.sp --destdir data_bin

fairseq-generate data_bin --path m39v1/checkpoint.pt --prefix-size 1 --remove-bpe sentencepiece --batch-size ${batch_size} > output.txt

python get_translations.py \
  --input-file output.txt \
  --language ${target} \
  --output-file ${output}
