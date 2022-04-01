# fmt: off
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import json
from fairseq.models.bart import BARTModel
import argparse

XSUM_KWARGS = dict(beam=6, lenpen=1.0, max_len_b=60, min_len=10, no_repeat_ngram_size=3)
CNN_KWARGS = dict(beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)


@torch.no_grad()
def generate(bart, infile, outfile="bart_hypo.txt", bsz=32, n_obs=None, nbest=1, **eval_kwargs):
    count = 1

    # if n_obs is not None: bsz = min(bsz, n_obs)

    with open(infile) as source, open(outfile, "w") as fout:
        sline = source.readline().strip()
        slines = [sline]
        for sline in source:
            if n_obs is not None and count > n_obs:
                break
            if count % bsz == 0:
                encoded = [bart.encode(line) for line in slines]
                batched_hypos = bart.generate(encoded, **eval_kwargs)
                for hypos in batched_hypos:
                    decoded = [
                        bart.decode(hypo["tokens"]) for hypo in hypos[:nbest]
                    ]
                    fout.write(json.dumps(decoded) + "\n")
                    fout.flush()

                slines = []

            slines.append(sline.strip())
            count += 1

        if slines != []:
            encoded = [bart.encode(line) for line in slines]
            batched_hypos = bart.generate(encoded, **eval_kwargs)
            for hypos in batched_hypos:
                decoded = [
                    bart.decode(hypo["tokens"]) for hypo in hypos[:nbest]
                ]
                fout.write(json.dumps(decoded) + "\n")
                fout.flush()


def main():
    """
    Usage::

         python examples/bart/summarize.py \
            --model-dir $HOME/bart.large.cnn \
            --model-file model.pt \
            --src $HOME/data-bin/cnn_dm/test.source
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        required=True,
        type=str,
        default="bart.large.cnn/",
        help="path containing model file and src_dict.txt",
    )
    parser.add_argument(
        "--model-file",
        default="checkpoint_best.pt",
        help="where in model_dir are weights saved",
    )
    parser.add_argument(
        "--src", default="test.source", help="text to summarize", type=str
    )
    parser.add_argument(
        "--out", default="test.hypo", help="where to save summaries", type=str
    )
    parser.add_argument("--bsz", default=32, help="where to save summaries", type=int)
    parser.add_argument(
        "--n", default=None, help="how many examples to summarize", type=int
    )
    parser.add_argument(
        "--xsum-kwargs",
        action="store_true",
        default=False,
        help="if true use XSUM_KWARGS else CNN_KWARGS",
    )
    parser.add_argument(
        "--nbest",
        default=1,
        type=int
    )
    parser.add_argument(
        "--beam-size",
        type=int
    )
    args = parser.parse_args()
    eval_kwargs = XSUM_KWARGS if args.xsum_kwargs else CNN_KWARGS
    if args.beam_size is not None:
        eval_kwargs["beam"] = args.beam_size

    if args.model_dir == "pytorch/fairseq":
        bart = torch.hub.load("pytorch/fairseq", args.model_file)
    else:
        bart = BARTModel.from_pretrained(
            args.model_dir,
            checkpoint_file=args.model_file,
            data_name_or_path=args.model_dir,
        )
    bart = bart.eval()
    if torch.cuda.is_available():
        bart = bart.cuda().half()
    generate(
        bart, args.src, bsz=args.bsz, n_obs=args.n, outfile=args.out, nbest=args.nbest, **eval_kwargs
    )


if __name__ == "__main__":
    main()
