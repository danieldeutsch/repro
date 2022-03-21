import argparse
import json
import numpy as np
from collections import defaultdict
from typing import List

from moverscore_v2 import get_idf_dict, word_mover_score

# Copied from the original repository:
# https://github.com/AIPHES/emnlp19-moverscore/blob/55e785bb3554b8f1c3e32525d97be1ee7bc23a76/examples/example.py
# Edited to accept stopwords
def sentence_score(hypothesis: str, references: List[str], stopwords: List[str]):
    idf_dict_hyp = defaultdict(lambda: 1.0)
    idf_dict_ref = defaultdict(lambda: 1.0)

    hypothesis = [hypothesis] * len(references)

    scores = word_mover_score(
        references,
        hypothesis,
        idf_dict_ref,
        idf_dict_hyp,
        stop_words=stopwords,
        n_gram=1,
        remove_subwords=False,
    )

    sentence_score = np.mean(scores)

    return sentence_score


def main(args):
    # Optionally load the stopwords
    if args.use_stopwords.lower() == "true":
        with open("stopwords.txt", "r", encoding="utf-8") as f:
            stopwords = set(f.read().strip().split(" "))
    else:
        stopwords = []

    with open(args.output_file, "w") as out:
        with open(args.input_file, "r") as f:
            for line in f:
                data = json.loads(line)
                candidate = data["candidate"]
                references = data["references"]

                score = sentence_score(candidate, references, stopwords)
                out.write(json.dumps({"moverscore": score}) + "\n")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--use-stopwords", required=True)
    argp.add_argument("--batch-size", type=int, default=48)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
