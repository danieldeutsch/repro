import argparse
import json

from bart_score import BARTScorer


def main(args):
    instances = []
    with open(args.input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            instances.append(data)

    sources = []
    targets = []
    for instance in instances:
        candidate = instance["candidates"]
        for reference in instance["references"]:
            sources.append(candidate)
            targets.append(reference)

    if args.device == -1:
        device = "cpu"
    else:
        device = f"cuda:{args.device}"

    metric = BARTScorer(device=device)
    if args.use_parabank.lower() == "true":
        metric.load(path="bart.pth")

    scores = metric.score(sources, targets, batch_size=args.batch_size)

    with open(args.output_file, "w") as out:
        index = 0
        for instance in instances:
            total = 0
            for _ in instance["references"]:
                total += scores[index]
                index += 1

            bartscore = total / len(instance["references"])
            out.write(json.dumps({"bartscore": bartscore}) + "\n")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--device", required=True, type=int)
    argp.add_argument("--batch-size", required=True, type=int)
    argp.add_argument("--use-parabank", required=True)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
