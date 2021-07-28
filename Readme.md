# Repro
![Master](https://github.com/danieldeutsch/repro/workflows/Master/badge.svg?branch=master&event=push)

Repro is a library for reproducing results from research papers.
For now, it is focused on making predictions with pre-trained models as easy as possible.

Currently, running pre-trained models can be difficult to do.
Some models require specific versions of dependencies, require complicated preprocessing steps, have their own input and output formats, are poorly documented, etc.

Repro addresses these problems by packaging each of the pre-trained models in its own Docker container, which includes the pre-trained models themselves as well as all of the code, dependencies, and environment setup required to run them.
Then, Repro provides lightweight Python code to read the input data, pass the data to a Docker container, run prediction in the container, and return the output to the user.
Since the complicated model-specific code is isolated within Docker, the user does not need to worry about setting up the environment correctly or know how the model is implemented at all.
**As long as you have a working Docker installation, then you can run every model included in repro with no additional effort.**
It should "just work" (at least that is the goal).

## Installation Instructions
First, you need to have a working Docker installation.
See [here](tutorials/docker.md) for installation instructions as well as scripts to verify your setup is working.

Then, we recommend creating a conda environment specific to repro before installing the library:
```shell script
conda create -n repro python=3.6
conda activate repro
```

For users:
```shell script
pip install repro
```

For developers:
```shell script
git clone https://github.com/danieldeutsch/repro
cd repro
pip install --editable .
```                                       

## Example Usage
Here is an example of how Repro can be used, highlighting how simple it is to run a complex model pipeline.
We will focus on generating summaries of a document with three different models: Liu & Lapata (2019) ([paper](https://arxiv.org/abs/1908.08345), [docs](models/liu2019/Readme.md)); Lewis et al. (2020) ([paper](https://arxiv.org/abs/1910.13461), [docs](models/lewis2020/Readme.md)); and Dou et al. (2021) ([paper](https://arxiv.org/abs/2010.08014), [docs](models/dou2021/Readme.md)).

First, you have to build the Docker images for each of the models, which is done through the `repro setup` command (run `repro setup <model-name> --help` to see more details).
Each command optionally marks which pre-trained models from the original papers are included in the Docker image.
This example only includes those trained on CNN/DailyMail:
```shell script
repro setup liu2019 --bertsumext-cnndm --bertsumextabs-cnndm
repro setup lewis2020 --cnndm
repro setup dou2021
```
Each command will take a few minutes to download and install all of the necessary dependencies in the Docker image, but you only have to do it once.

Now, all you have to do is instantiate the classes and run `predict`:

```python
from repro.models.liu2019 import BertSumExtAbs
from repro.models.lewis2020 import BART
from repro.models.dou2021 import SentenceGSumModel

# Each of these classes uses the pre-trained weights that we want to use
# by default, but you can specify others if you want to
liu2019 = BertSumExtAbs()
lewis2020 = BART()
dou2021 = SentenceGSumModel()

# Here's the document we want to summarize (it's not very long,
# but you get the point)
document = (
    "Joseph Robinette Biden Jr. was elected the 46th president of the United States "
    "on Saturday, promising to restore political normalcy and a spirit of national "
    "unity to confront raging health and economic crises, and making Donald J. Trump "
    "a one-term president after four years of tumult in the White House."
)

# Now, run `predict` to generate the summaries from the models
summary1 = liu2019.predict(document)
summary2 = lewis2020.predict(document)
summary3 = dou2021.predict(document)
```

Behind the scenes, Repro is running each model in its own Docker container.
The Liu & Lapata (2019) model is tokenizing and sentence splitting the input document with Stanford CoreNLP, then running BERT with `torch==1.1.0` and `transformers==1.2.0`.
The Lewis et al. (2020) is running the original BART code in `fairseq` with `torch==1.9.0`, and the Dou et al. (2021) model is running its own fork of the BART code with `torch==1.4.0` and calling a model from Liu & Lapata (2019) as a subroutine.
**But you don't need to know about any of that to run the models!**
All of the complex logic and environment details are taken care of by the Docker container, so all you have to do is call `predict()`.
It's that simple!

## Models Implemented in Repro
See the [`models`](models) directory or [this file](Papers.md) to see the list of papers with models currently supported by Repro.
Each model contains information in its Readme about how to use it as well as whether or not it currently reproduces the results reported in its respective paper or if it hasn't been tested yet.
If it has been tested, the code to reproduce the results is also included.
