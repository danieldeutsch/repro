import argparse
import json
import os
from sacrebleu import sentence_bleu


def main(args):
    dirname = os.path.dirname(args.output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(args.output_file, "w") as out:
        with open(args.input_file, "r") as f:
            for line in f:
                data = json.loads(line)
                candidate = data["candidate"]
                references = data["references"]
                result = sentence_bleu(candidate, references)
                out.write(json.dumps({"sentbleu": result.score}) + "\n")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
