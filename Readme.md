# Repro
![Master](https://github.com/danieldeutsch/repro/workflows/Master/badge.svg?branch=master&event=push)

Repro is a library for reproducing results from research papers.
For now, it is focused on making predictions with pre-trained models as easy as possible.

Currently, running pre-trained models can be difficult to do.
Some models require specific versions of dependencies, have their own input and output formats, are poorly documented, etc.

Repro addresses these problems by packaging each of the pre-trained models in its own Docker container, which includes the pre-trained models themselves as well as all of the code and dependencies required to run them.
Then, repro provides lightweight Python code to read the input data, pass the data to a Docker container with a model-specific input format, run prediction in the container, and return the output to the user.
Since the complicated model-specific code is isolated within Docker, the user does not need to worry about setting up the environment correctly or know how the model is implemented at all.
As long as you have a working Docker installation, then you can run every model included in repro with no additional effort. 
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

# Here's the document we want to summarize
document = ""

# Now, run `predict` to generate the summaries from the models
summary1 = liu2019.predict(document)
summary2 = lewis2020.predict(document)
summary3 = dou2021.predict(document)
```
Behind the scenes, the Liu & Lapata (2019) model is tokenizing and sentence splitting the input document with Stanford CoreNLP, then running BERT with `torch==1.1.0` and `transformers==1.2.0`.
The Lewis et al. (2020) is running the original BART code in `fairseq` with `torch==1.9.0`, and the Dou et al. (2021) model is running its own fork of the BART code with `torch==1.4.0` and calling a model from Liu & Lapata (2019) as a subroutine.
But you don't need to know about any of that to run the models!
It is that simple.
