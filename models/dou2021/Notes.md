This file contains information that would be relevant if this model is ever extended to implement training functionality.

Training the model requires preprocessing the data with the roberta BPE encoder.
The guided_summarization repository has its own directory called "fairseq", but it does not have the "examples/roberta/multiprocessing_bpe_encoder.py code to do the preprocessing, so you need to use the latest fairseq code.
You also need to download data files.
```
RUN git clone https://github.com/pytorch/fairseq
RUN cd fairseq && git checkout c1624b273b206cc7c0a1529be4d2f35b38607ec5
```

Then, following the instructions in the GSum repo and [here](https://github.com/pytorch/fairseq/blob/67ff6baa42c1208d0da85f5af2f01689034d1dfd/examples/bart/README.summarization.md), download data and run the BPE.
There should be input files `{train,val}.{source,target,z}`.
Each file has 1 document, summary, or supervision signal per line.
```
mkdir -p bpe
cd fairseq
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json' 
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

for split in train val; do
  for inp in source target z; do
    INPUT=../${split}.${inp}
    OUTPUT=../bpe/${split}.bpe.${inp}
    python -m examples.roberta.multiprocessing_bpe_encoder \
      --encoder-json encoder.json \
      --vocab-bpe vocab.bpe \
      --inputs "$INPUT" \
      --outputs "$OUTPUT" \
      --workers 60 \
      --keep-empty
  done
done
```

Then the encoded files need to be binarized by the GSum code
```
mkdir bin
cd guided_summarization/bart
BPE_DIR=../../bpe
BIN_DIR=../../bin
python -m fairseq_cli.guided_preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref $BPE_DIR"/train.bpe" \
  --validpref $BPE_DIR"/val.bpe" \
  --destdir $BIN_DIR \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
```

The encoding/binarization is not necessary for running inference on the test data.