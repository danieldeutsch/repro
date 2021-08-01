import argparse
import json
import os
from sacrebleu import corpus_bleu


def main(args):
    # Load the data
    candidates = []
    references_list = []
    with open(args.input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            candidates.append(data["candidate"])
            references_list.append(data["references"])

    # sacrebleu requires an equal number of references for
    # each candidate, but you can use `None` for padding
    max_references = max(len(references) for references in references_list)
    for references in references_list:
        references.extend([None] * (max_references - len(references)))

    # The references are expected to be inverted. That is, references_list[i][j]
    # is the i-th reference for the j-th candidate
    inverted_references_list = [[] for _ in range(max_references)]
    for references in references_list:
        for i, reference in enumerate(references):
            inverted_references_list[i].append(reference)

    # Calculate BLEU
    result = corpus_bleu(candidates, inverted_references_list)

    # Save the result
    dirname = os.path.dirname(args.output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(args.output_file, "w") as out:
        out.write(json.dumps({"bleu": result.score}))


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
