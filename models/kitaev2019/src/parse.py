import argparse
import benepar
import json
import spacy


def main(args):
    nlp = spacy.load("en_core_web_md")
    if spacy.__version__.startswith("2"):
        nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
    else:
        nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    texts = []
    with open(args.input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            text = data["text"]
            texts.append(text)

    with open(args.output_file, "w") as out:
        for doc in nlp.pipe(texts):
            parses = [sent._.parse_string for sent in doc.sents]
            out.write(json.dumps({"parses": parses}) + "\n")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--model", required=True)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
