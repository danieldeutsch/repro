## Description
This experiment runs the FactCC and FactCCX models on the validation and testing data collected in the paper and compares the results to those reported in the paper and original code.

## Usage
Running the evaluation requires the following extra Python libraries:
```
gdown
scikit-learn==0.24.2
```

Then execute these scripts from this directory.

First, download the CNN/DailyMail data and the annotations released by the paper:
```shell script
gdown https://drive.google.com/uc?id=0BwmD_VLjROrfTHk4NFg2SndKcjQ --output cnn_stories.tgz
gdown https://drive.google.com/uc?id=0BwmD_VLjROrfM1BxdkxVaTY2bWs --output dailymail_stories.tgz

wget https://storage.googleapis.com/sfr-factcc-data-research/unpaired_annotated_data.tar.gz
tar -xvf unpaired_annotated_data.tar.gz
```

Then score the data:
```shell script
python run.py \
    --cnn-tar cnn_stories.tgz \
    --dailymail-tar dailymail_stories.tgz \
    --data-dir unpaired_annotated_data \
    --device 1 \
    --output-file output.json
```

## Results
The pretrained model directories contain the expected balanced accuracy and micro-F1 scores of each model on both dataset splits.
We calculate those values as well as the F1 scores reported in the paper:

FactCC
||Split|Bal-Acc|Micro-F1|F1|
|-|-|-|-|-|
|Reported|Valid|0.761|0.861|-|
|Ours|Valid|0.761|0.861|-|
|Reported|Test|0.727|0.861|0.5106|
|Ours|Test|0.727|0.861|0.493|

The balanced accuracy scores on the test split are slightly lower than the 74.15 reported in the paper.

FactCCX
||Split|Bal-Acc|Micro-F1|F1|
|-|-|-|-|-|
|Reported|Valid|0.754|0.859|-|
|Ours|Valid|0.754|0.859|-|
|Reported|Test|0.729|0.865|0.5005|
|Ours|Test|0.729|0.865|0.5000|

Overall, these results match exactly what is expected from the code, so it is a faithful implementation of the metrics.