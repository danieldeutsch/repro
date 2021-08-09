import argparse
import json

from summaqa import QA_Metric, QG_masked


def main(args):
    question_generator = QG_masked()
    qa_metric = QA_Metric()

    with open(args.output_file, "w") as out:
        with open(args.input_file, "r") as f:
            for line in f:
                data = json.loads(line)
                candidate = data["candidate"]
                source = data["source"]

                masked_questions, answer_spans = question_generator.get_questions(
                    source
                )
                score = qa_metric.compute(masked_questions, answer_spans, candidate)
                out.write(json.dumps({"summaqa": score}) + "\n")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
