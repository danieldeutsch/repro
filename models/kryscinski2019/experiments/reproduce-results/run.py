import argparse
import json
import tarfile
from collections import defaultdict
from sklearn.metrics import balanced_accuracy_score, f1_score
from tqdm import tqdm
from typing import List, Tuple

from repro.common.io import read_jsonl_file
from repro.models.kryscinski2019 import FactCC, FactCCX


def parse_story_file(content):
    """
    Remove article highlights and unnecessary white characters.
    """
    content_raw = content.split("@highlight")[0]
    content = " ".join(filter(None, [x.strip() for x in content_raw.split("\n")]))
    return content


def evaluate(labels: List[int], predictions: List[int]) -> Tuple[float, float, float]:
    # Calculates the same metrics included in the model evaluations (balanced accuracy and micro F1)
    # plus a standard F1
    assert len(labels) == len(predictions)
    bacc = balanced_accuracy_score(y_true=labels, y_pred=predictions)
    micro_f1 = f1_score(y_true=labels, y_pred=predictions, average="micro")
    f1 = f1_score(y_true=labels, y_pred=predictions)
    return bacc, micro_f1, f1


def main(args):
    factcc = FactCC()
    factccx = FactCCX()

    results_dict = defaultdict(dict)
    with tarfile.open(args.cnn_tar, "r") as cnn_tar:
        with tarfile.open(args.dailymail_tar, "r") as dailymail_tar:
            for split in ["val", "test"]:
                instances = read_jsonl_file(f"{args.data_dir}/{split}/data-dev.jsonl")
                inputs = []
                labels = []
                for instance in tqdm(instances):
                    candidate = instance["claim"]

                    label = 0 if instance["label"] == "CORRECT" else 1
                    labels.append(label)

                    # Remove "cnndm/"
                    filepath = instance["filepath"]
                    filepath = filepath[6:]
                    if filepath.startswith("cnn"):
                        content = cnn_tar.extractfile(f"./{filepath}").read().decode()
                    else:
                        content = (
                            dailymail_tar.extractfile(f"./{filepath}").read().decode()
                        )
                    source = parse_story_file(content)

                    inputs.append({"candidate": candidate, "sources": [source]})

                _, factcc_results = factcc.predict_batch(inputs)
                predictions = [results["factcc"]["label"] for results in factcc_results]
                acc, micro_f1, f1 = evaluate(labels, predictions)
                results_dict["factcc"][split] = {
                    "bacc": acc,
                    "micro_f1": micro_f1,
                    "f1": f1,
                }

                _, factccx_results = factccx.predict_batch(inputs)
                predictions = [
                    results["factccx"]["label"] for results in factccx_results
                ]
                acc, micro_f1, f1 = evaluate(labels, predictions)
                results_dict["factccx"][split] = {
                    "bacc": acc,
                    "micro_f1": micro_f1,
                    "f1": f1,
                }

    with open(args.output_file, "w") as out:
        out.write(json.dumps(results_dict, indent=2))


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--cnn-tar", required=True)
    argp.add_argument("--dailymail-tar", required=True)
    argp.add_argument("--data-dir", required=True)
    argp.add_argument("--device", required=True, type=int)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
