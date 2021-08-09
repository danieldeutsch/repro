import argparse
import json

from nubia_score import Nubia


def main(args):
    metric = Nubia()
    six_dim = args.six_dim.lower() == "true"

    with open(args.output_file, "w") as out:
        with open(args.input_file, "r") as f:
            for line in f:
                data = json.loads(line)
                candidate = data["candidate"]
                reference = data["reference"]

                scores = metric.score(
                    reference,
                    candidate,
                    get_features=True,
                    six_dim=six_dim,
                    aggregator=args.aggregator,
                )
                out.write(json.dumps({"nubia": scores}) + "\n")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--six-dim", required=True)
    argp.add_argument("--aggregator", required=True)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
