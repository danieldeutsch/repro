if [ "$CNNDM" = "true" ]; then
  wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz
  tar -xvf bart.large.cnn.tar.gz
  rm bart.large.cnn.tar.gz
fi

if [ "$XSUM" = "true" ]; then
  wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.xsum.tar.gz
  tar -xvf bart.large.xsum.tar.gz
  rm bart.large.xsum.tar.gz
fi