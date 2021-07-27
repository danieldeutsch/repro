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
