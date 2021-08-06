VERSION = "1.0"
MODEL_NAME = "deutsch2021"
DOCKERHUB_REPRO = f"danieldeutsch/{MODEL_NAME}"
DEFAULT_IMAGE = f"{DOCKERHUB_REPRO}:{VERSION}"
CITATION = """
@misc{deutsch2021questionanswering,
      title={{Towards Question-Answering as an Automatic Metric for Evaluating the Content Quality of a Summary}}, 
      author={Daniel Deutsch and Tania Bedrax-Weiss and Dan Roth},
      year={2021},
      eprint={2010.00490},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""
