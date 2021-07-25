import argparse
import json
import os

from qaeval import QuestionGenerationModel


def main(args):
    inputs = []
    with open(args.input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            inputs.append((data["context"], data["start"], data["end"]))

    model = QuestionGenerationModel(args.model_file, args.cuda_device, args.batch_size)
    questions = model.generate_all(inputs)

    dirname = os.path.dirname(args.output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(args.output_file, "w") as out:
        for question in questions:
            out.write(json.dumps({"question": question}) + "\n")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--model-file", required=True)
    argp.add_argument("--cuda-device", type=int, required=True)
    argp.add_argument("--batch-size", type=int, required=True)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
