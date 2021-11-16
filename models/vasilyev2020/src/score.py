import argparse
import json
import os
from blanc import BlancHelp, BlancTune


def main(args):
    kwargs = json.loads(args.kwargs)
    device = "cpu" if args.device == -1 else "cuda"

    if args.type == "tune":
        blanc = BlancTune(device=device, random_seed=args.random_seed, **kwargs)
    elif args.type == "help":
        blanc = BlancHelp(device=device, **kwargs)
    else:
        raise Exception(f"Unknown BLANC type: {args.type}")

    documents = []
    summaries_list = []
    with open(args.input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            documents.append(data["document"])
            summaries_list.append(data["summaries"])

    scores_list = blanc.eval_summaries_for_docs(documents, summaries_list)

    dirname = os.path.dirname(args.output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(args.output_file, "w") as out:
        out.write(json.dumps(scores_list))


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--type", required=True, choices=["help", "tune"])
    argp.add_argument("--device", required=True, type=int)
    argp.add_argument("--random-seed", required=True, type=int)
    argp.add_argument("--kwargs", required=True)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
