import argparse
import json
import numpy as np

from moverscore_v2 import get_idf_dict, word_mover_score


def main(args):
    candidates = []
    references_list = []

    unique_candidates = set()
    unique_references = set()

    with open(args.input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            candidate = data["candidate"]
            references = data["references"]
            candidates.append(candidate)
            references_list.append(references)
            unique_candidates.add(candidate)
            for reference in references:
                unique_references.add(reference)

    # Calculate the idf dicts on the unique texts
    idf_dict_candidates = get_idf_dict(list(unique_candidates))
    idf_dict_references = get_idf_dict(list(unique_references))

    # Optionally load the stopwords
    if args.use_stopwords.lower() == "true":
        with open("stopwords.txt", "r", encoding="utf-8") as f:
            stopwords = set(f.read().strip().split(" "))
    else:
        stopwords = []

    # Flatten the candidates and references to be passed through `word_mover_score` all
    # at once for faster processing
    flat_candidates = []
    flat_references = []
    for candidate, references in zip(candidates, references_list):
        for reference in references:
            flat_candidates.append(candidate)
            flat_references.append(reference)

    scores = word_mover_score(
        flat_references,
        flat_candidates,
        idf_dict_references,
        idf_dict_candidates,
        stop_words=stopwords,
        n_gram=1,
        remove_subwords=True,
        batch_size=args.batch_size,
    )

    with open(args.output_file, "w") as out:
        # Take the mean over the references for the candidate
        index = 0
        for references in references_list:
            this_scores = []
            for _ in references:
                this_scores.append(scores[index])
                index += 1
            average_score = np.mean(this_scores)
            out.write(json.dumps({"moverscore": average_score}) + "\n")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--use-stopwords", required=True)
    argp.add_argument("--batch-size", type=int, default=48)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
