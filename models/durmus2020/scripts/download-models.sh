# Download the question-generation model
mkdir -p feqa/bart_qg/checkpoints
gdown https://drive.google.com/uc?id=1GFnimonLFgGal1LT6KRgMJZLbxmNJvxF --output feqa/bart_qg/checkpoints/checkpoint_best.pt
gdown https://drive.google.com/uc?id=17CShx4cUEQTl_gpLapnbMsc7CmDAaV7r --output feqa/bart_qg/checkpoints/dict.src.txt
gdown https://drive.google.com/uc?id=1_dUN7CQZdqPxoiezzWp5yByuEXVJFwce --output feqa/bart_qg/checkpoints/dict.tgt.txt

# Download the QA model
mkdir -p feqa/qa_models/squad1.0
gdown https://drive.google.com/uc?id=1IwWhQf9MP2G-vOBsQD87kMMEBS0IvcXa --output feqa/qa_models/squad1.0/config.json
gdown https://drive.google.com/uc?id=1tsWhCsXSxxgkBMBnGB9wkOliJH8K3Prs --output feqa/qa_models/squad1.0/dev-v1.1.json
gdown https://drive.google.com/uc?id=1p-LlVVAGuMYjFckjK5HxdiK5xEuM-2Ev --output feqa/qa_models/squad1.0/evaluate-v1.1.py
gdown https://drive.google.com/uc?id=1pWMsSTTwcoX0l75bzNFjvSC7firawp9M --output feqa/qa_models/squad1.0/pytorch_model.bin
gdown https://drive.google.com/uc?id=1yZKNFU7md4KPGmThPwsp4dt95HkKsArX --output feqa/qa_models/squad1.0/run_squad.py
gdown https://drive.google.com/uc?id=1rbv75oE5x0rXxtGGXETTvLBoHK5h3Lfj --output feqa/qa_models/squad1.0/special_tokens_map.json
gdown https://drive.google.com/uc?id=1oPM62qOWofGnaLmlX_CWkYKbZ-KEMtym --output feqa/qa_models/squad1.0/tokenizer_config.json
gdown https://drive.google.com/uc?id=1y9_EgnoBbm0SJeCaNZFfjOyraeA-qfqP --output feqa/qa_models/squad1.0/train-v1.1.json
gdown https://drive.google.com/uc?id=1r49Y1Cp2t6_II2xjOyxbvYVvp2EQj3zu --output feqa/qa_models/squad1.0/training_args.bin
gdown https://drive.google.com/uc?id=1iGZrP6_3PiiH0pcF4zoSbqAsWdFvimfF --output feqa/qa_models/squad1.0/vocab.txt
