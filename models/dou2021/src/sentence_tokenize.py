import argparse
import os
import spacy
from tqdm import tqdm


def main(args):
    nlp = spacy.load("en_core_web_sm")

    dirname = os.path.dirname(args.output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(args.output_file, "w") as out:
        with open(args.input_file, "r") as f:
            for line in tqdm(f, desc="Running sentence splitting"):
                line = line.strip()
                sentences = [str(sentence) for sentence in nlp(line).sents]
                out.write("<q>".join(sentences) + "\n")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("input_file")
    argp.add_argument("output_file")
    args = argp.parse_args()
    main(args)
