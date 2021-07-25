import argparse
import json
import os
from nltk.tokenize import sent_tokenize


def main(args):
    dirname = os.path.dirname(args.output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(args.output_file, "w") as out:
        with open(args.input_file, "r") as f:
            for line in f:
                data = json.loads(line)
                text = data["text"]
                sentences = [sentence for sentence in sent_tokenize(text)]
                out.write(json.dumps({"sentences": sentences}) + "\n")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("input_file")
    argp.add_argument("output_file")
    args = argp.parse_args()
    main(args)
