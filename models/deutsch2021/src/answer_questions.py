import argparse
import json
import os

from qaeval import QuestionAnsweringModel


def main(args):
    inputs = []
    with open(args.input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            inputs.append((data["question"], data["context"]))

    model = QuestionAnsweringModel(args.model_dir, args.cuda_device, args.batch_size)
    answers_dicts = model.answer_all(
        inputs, return_offsets=True, try_fixing_offsets=True, return_dicts=True
    )

    dirname = os.path.dirname(args.output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(args.output_file, "w") as out:
        for answers_dict in answers_dicts:
            out.write(
                json.dumps(
                    {
                        "prediction": answers_dict["prediction"],
                        "probability": answers_dict["probability"],
                        "null_probability": answers_dict["null_probability"],
                        "start": answers_dict["start"],
                        "end": answers_dict["end"],
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--model-dir", required=True)
    argp.add_argument("--cuda-device", type=int, required=True)
    argp.add_argument("--batch-size", type=int, required=True)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
