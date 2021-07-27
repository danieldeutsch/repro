import argparse
import json
import os
from allennlp.predictors import Predictor
from typing import Dict, List

from lerc.lerc_predictor import LERCPredictor


def main(args):
    # Loads an AllenNLP Predictor that wraps our trained model
    predictor = Predictor.from_path(
        archive_path=args.model_path,
        predictor_name="lerc",
        cuda_device=args.cuda_device,
    )

    dirname = os.path.dirname(args.output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(args.output_file, "w") as out:
        with open(args.input_file, "r") as f:
            batch = []
            for line in f:
                data = json.loads(line)
                batch.append(data)

                if len(batch) == args.batch_size:
                    predictions = predictor.predict_batch_json(batch)
                    for prediction in predictions:
                        out.write(json.dumps(prediction) + "\n")
                    batch = []

            if len(batch) > 0:
                predictions = predictor.predict_batch_json(batch)
                for prediction in predictions:
                    out.write(json.dumps(prediction) + "\n")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--model-path", required=True)
    argp.add_argument("--batch-size", required=True, type=int)
    argp.add_argument("--cuda-device", required=True, type=int)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
