import argparse
import json

from repro.models.goyal2020 import DAE


def main(args):
    data = json.load(open(args.input_file, "r"))
    correct_inputs = []
    incorrect_inputs = []
    for instance in data:
        correct_inputs.append(
            {
                "candidate": instance["correct_sent"],
                "sources": [instance["article_sent"]],
            }
        )
        incorrect_inputs.append(
            {
                "candidate": instance["incorrect_sent"],
                "sources": [instance["article_sent"]],
            }
        )

    results = {}
    for name in ["dae_basic", "dae_w_syn", "dae_w_syn_hallu"]:
        model = DAE(model=name, device=args.device)
        _, correct_scores = model.predict_batch(correct_inputs)
        _, incorrect_scores = model.predict_batch(incorrect_inputs)
        total = len(correct_scores)
        num_correct = 0
        for correct, incorrect in zip(correct_scores, incorrect_scores):
            if correct["dae"] > incorrect["dae"]:
                num_correct += 1
        accuracy = num_correct / total
        results[name] = accuracy

    with open(args.output_file, "w") as out:
        out.write(json.dumps(results, indent=2))


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--device", required=True, type=int)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
