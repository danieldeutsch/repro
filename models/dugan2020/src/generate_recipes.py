"""
Adapted from https://github.com/kirubarajan/roft/blob/master/generation/interactive_test.py to
process a batch of inputs.
"""
import argparse
import json
import numpy as np
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(args):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    if torch.cuda.is_available():
        model = model.cuda()

    dirname = os.path.dirname(args.output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(args.output_file, "w") as out:
        with open(args.input_file, "r") as f:
            for line in f:
                data = json.loads(line)
                name = data["name"]
                ingredients = "\n".join(data["ingredients"])

                input_text = f"HOW TO MAKE: {name}\nIngredients:\n{ingredients}."
                input_tensor = tokenizer.encode(input_text, return_tensors="pt").to(
                    model.device
                )
                outputs = model.generate(
                    input_tensor,
                    do_sample=True,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    pad_token_id=tokenizer.eos_token_id,
                    max_length=args.max_length,
                )
                recipe = [tokenizer.decode(x) for x in outputs][0]
                out.write(json.dumps({"recipe": recipe}) + "\n")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--model-name", required=True)
    argp.add_argument("--top-p", type=float, default=0.7)
    argp.add_argument("--repetition-penalty", type=float, default=1.2)
    argp.add_argument("--max-length", type=int, default=256)
    argp.add_argument("--random-seed", type=int, default=4)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
