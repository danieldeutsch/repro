import argparse
import json

from qaeval import QAEval


def main(args):
    kwargs = json.loads(args.kwargs)
    metric = QAEval(
        generation_model_path="models/generation/model.tar.gz",
        answering_model_dir="models/answering",
        lerc_model_path="models/lerc/model.tar.gz",
        lerc_pretrained_model_path="models/lerc/pretraining.tar.gz",
        **kwargs
    )

    candidates = []
    references_list = []
    with open(args.input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            candidates.append(data["candidate"])
            references_list.append(data["references"])

    results = metric.score_batch(candidates, references_list, return_qa_pairs=True)

    with open(args.output_file, "w") as out:
        for metrics, qa_pairs in results:
            out.write(json.dumps({"metrics": metrics, "qa_pairs": qa_pairs}) + "\n")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--kwargs", required=True)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
