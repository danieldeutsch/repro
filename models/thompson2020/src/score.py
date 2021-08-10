import argparse
import json

from prism import Prism


def main(args):
    metric = Prism(model_dir="../m39v1", lang=args.language)

    candidates = []
    references = []
    sources = []

    with open(args.input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            candidates.append(data["candidate"])
            references.append(data["reference"])
            sources.append(data["source"])

    # Input error checking is done in the model code, so we don't need to check here.
    # Assume that if there are references, they all have references and no sources.
    has_references = any(reference is not None for reference in references)
    if has_references:
        scores = metric.score(cand=candidates, ref=references, segment_scores=True)
    else:
        # Assume all have sources
        scores = metric.score(cand=candidates, src=sources, segment_scores=True)

    with open(args.output_file, "w") as out:
        for score in scores:
            out.write(json.dumps({"prism": float(score)}) + "\n")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--language", required=True)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
