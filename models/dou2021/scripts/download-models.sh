mkdir bart_sentence
cd bart_sentence
gdown https://drive.google.com/uc?id=1BMKhAh2tG5p8THxugZWMPc7NXqwJDHLw
mv bart_sentence.pt model.pt
wget https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
mv dict.txt dict.source.txt
cp dict.source.txt dict.target.txt
