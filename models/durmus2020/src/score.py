import argparse
import json

from feqa import FEQA


def main(args):
    cuda = args.cuda_device != -1
    metric = FEQA(use_gpu=cuda)
    metric.batch_size = args.batch_size

    candidates = []
    sources = []
    with open(args.input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            candidates.append(data["candidate"])
            sources.append(data["source"])

    scores = metric.compute_score(sources, candidates, aggregate=False)

    with open(args.output_file, "w") as out:
        for score in scores:
            out.write(json.dumps({"feqa": score}) + "\n")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--cuda-device", required=True, type=int)
    argp.add_argument("--batch-size", required=True, type=int)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
