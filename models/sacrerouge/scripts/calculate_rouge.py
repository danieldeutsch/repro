import argparse
import json
import os

from repro.common.io import read_jsonl_file
from repro.models.sacrerouge import ROUGE


def main(args):
    instances = read_jsonl_file(args.input_file)
    inputs = []
    for instance in instances:
        summary = instance["summary"]["text"]
        if "reference" in instance:
            references = [instance["reference"]["text"]]
        else:
            references = [reference["text"] for reference in instance["references"]]
        inputs.append({"summary": summary, "references": references})

    metric = ROUGE()
    scores = metric.predict_batch(inputs)

    dirname = os.path.dirname(args.output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(args.output_file, "w") as out:
        out.write(json.dumps(scores, indent=2))


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
