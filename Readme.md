# Repro
Repro is a library for reproducing results from research papers.
For now, it is focused on making predictions with pre-trained models as easy as possible.

Currently, running pre-trained models can be difficult to do.
Some models require specific versions of dependencies, have their own input and output formats, are poorly documented, etc.

Repro addresses these problems by packaging each of the pre-trained models in its own Docker container, which includes the pre-trained models themselves as well as all of the code and dependencies required to run them.
Then, repro provides lightweight Python code to read the input data, pass the data to a Docker container with a model-specific input format, run prediction in the container, and return the output to the user.
Each common NLP task (e.g., question-answering or summarization) has a standard interface, so running prediction with multiple models requires almost no extra work for the user. 
As long as you have a working Docker installation, you do not need to spend time trying to set up the pre-trained models or any of their dependencies.
It should "just work" (at least that is the goal).

## Installation Instructions
First, you need to have a working Docker installation (instructions coming soon).

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