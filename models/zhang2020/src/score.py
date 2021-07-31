import argparse
import json
import os

import bert_score


def main(args):
    candidates = []
    references_list = []

    with open(args.input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            candidates.append(data["candidate"])
            references_list.append(data["references"])

    device = None if args.cuda_device == -1 else args.cuda_device
    precisions, recall, f1s = bert_score.score(
        candidates,
        references_list,
        model_type=args.model_name,
        device=device,
        batch_size=args.batch_size,
        lang=args.language,
    )

    dirname = os.path.dirname(args.output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(args.output_file, "w") as out:
        for precision, recall, f1 in zip(precisions, recall, f1s):
            out.write(
                json.dumps(
                    {
                        "precision": precision.item(),
                        "recall": recall.item(),
                        "f1": f1.item(),
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--model-name")
    argp.add_argument("--cuda-device", required=True, type=int)
    argp.add_argument("--batch-size", type=int, default=64)
    argp.add_argument("--language")
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
