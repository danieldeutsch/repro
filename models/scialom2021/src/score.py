import argparse
import json
from tqdm import tqdm

from questeval.questeval_metric import QuestEval


def main(args):
    kwargs = json.loads(args.kwargs)
    is_cuda = kwargs.pop("isCuda", args.cuda_device != -1)
    metric = QuestEval(isCuda=is_cuda, **kwargs)

    with open(args.output_file, "w") as out:
        with open(args.input_file, "r") as f:
            for line in tqdm(f, desc="Scoring"):
                data = json.loads(line)
                candidate = data["candidate"]
                source = data["source"]
                reference = data["reference"]

                scores = metric.compute_all(
                    candidate, source=source, reference=reference
                )
                out.write(json.dumps(scores) + "\n")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--kwargs", required=True)
    argp.add_argument("--cuda-device", required=True, type=int)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
