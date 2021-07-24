import argparse
import json
import spacy


def main(args):
    nlp = spacy.load("en_core_web_sm")
    with open(args.output_file, "w") as out:
        with open(args.input_file, "r") as f:
            for line in f:
                data = json.loads(line)
                summary = data["summary"]["text"]
                sentences = [str(sentence) for sentence in nlp(summary).sents]
                data["summary"]["text"] = sentences
                out.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
