# fmt: off
#!/usr/bin/env python
import os
import argparse
import torch
import json

from bary_score import BaryScoreMetric
from depth_score import DepthScoreMetric
from infolm import InfoLM
from tqdm import tqdm
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")
    ####################
    # Common Arguments #
    ####################
    parser = argparse.ArgumentParser("Calculate Metrics Based on Statistical Measures of Similarity")
    parser.add_argument("--metric_name", type=str, default="baryscore",
                        choices=['depthscore', 'baryscore', 'infolm'], help=" which metric to compute")
    parser.add_argument(
        "-m", "--model", default='bert-base-uncased',
        help="BERT model name (default: bert-base-uncased) or path to a pretrain model",
    )
    parser.add_argument("-b", "--batch_size", type=int, default=3, help="batch size (default: 64)")
    parser.add_argument("-r", "--ref", type=str, required=True, help="reference file path or a string")
    parser.add_argument(
        "-c", "--cand", type=str, required=True, help="candidate (system outputs) file path or a string",
    )
    parser.add_argument("--idf", action="store_true", help="if true use idf")
    parser.add_argument("--output_file", required=True)
    ###################
    # Argument InfoLM #
    ###################
    parser.add_argument("--measure_to_use", type=str, default="fisher_rao",
                        choices=['kl', 'alpha', 'renyi', 'beta', 'ab', 'l1', "l2", "linf", 'fisher_rao'],
                        help=" which measure of information to use")
    parser.add_argument("--temperature", type=float, default=0.5, help=" temperature to calibrate the LM")
    parser.add_argument("--beta", type=float, default=0.5, help="beta parameter in the ab or beta div")
    parser.add_argument("--alpha", type=float, default=0.5, help="alpha parameter in the ab, alpha or renyi div")

    ######################
    # Argument BaryScore #
    ######################
    parser.add_argument("--last_layers", type=int, default=5, help="use the last  N layer in BERT")
    parser.add_argument("--sinkhorn_ref", type=float, default=0.01, help="weight of the KL in the SD")

    #######################
    # Argument DepthScore #
    #######################
    parser.add_argument("--layers_to_consider", type=int, default=9, help="use the N-th layer in BERT")
    parser.add_argument("--considered_measure", type=str, default='ai_irw',
                        choices=["irw", "ai_irw", "wasserstein", "sliced", "mmd"], help="")
    parser.add_argument("--p", type=float, default=5, help="the power of the ground cost")
    parser.add_argument("--eps", type=float, default=0.3, help="the highest level set")
    parser.add_argument("--n_alpha", type=float, default=5,
                        help="the Monte-Carlo parameter for the approximation of the integral  over alpha.")

    args = parser.parse_args()
    logging.info("Computing Score for {}".format(args.metric_name))
    if os.path.isfile(args.cand):
        with open(args.cand) as f:
            cands = [line.strip() for line in f]

        if os.path.exists(args.ref):
            with open(args.ref) as f:
                refs = [line.strip() for line in f]
        logging.info("Files opened Starting To Process")
    else:
        logging.info("Single Sentence Score")
        cands = [args.cand]
        refs = [args.ref]
        assert not args.idf, "do not support idf mode for a single pair of sentences"
    # do for loops :)
    if args.metric_name == 'infolm':
        metric = InfoLM(model_name=args.model, temperature=args.temperature, measure_to_use=args.measure_to_use,
                        use_idf_weights=args.idf, alpha=args.alpha, beta=args.beta)
    elif args.metric_name == 'depthscore':
        metric = DepthScoreMetric(model_name=args.model, layers_to_consider=args.layers_to_consider,
                                  considered_measure=args.considered_measure, p=args.p,
                                  eps=args.eps, n_alpha=args.n_alpha)
    elif args.metric_name == 'baryscore':
        metric = BaryScoreMetric(model_name=args.model, last_layers=args.last_layers, use_idfs=args.idf,
                                 sinkhorn_ref=args.sinkhorn_ref)
    else:
        raise NotImplementedError
    all_preds = []
    metric.prepare_idfs(cands, refs)
    batched_candidates = [cands[i:i + args.batch_size] for i in range(0, len(cands), args.batch_size)]
    batched_references = [refs[i:i + args.batch_size] for i in range(0, len(refs), args.batch_size)]
    for golden_batch, candidate_batch in tqdm(zip(batched_references, batched_candidates), 'Evaluation in Progress'):
        preds = metric.evaluate_batch(candidate_batch, golden_batch)
        all_preds.append(preds)

    # Rearrange for per-input dicts
    per_input_preds = []
    metric_name = list(preds.keys())[0]
    for batch in all_preds:
        num_instances = len(batch[metric_name])
        for i in range(num_instances):
            per_input_preds.append(
                {args.metric_name: {k: batch[k][i] for k in preds.keys()}}
            )

    with open(args.output_file, "w") as out:
        out.write(json.dumps(per_input_preds))

    # for k in preds.keys():
    #     l_preds = []
    #     for pred in all_preds:
    #         l_preds.append(pred[k])
    #     logging.info('Metric {} : {}'.format(k, sum(sum(l_preds, [])) / len(sum(l_preds, []))))


if __name__ == "__main__":
    main()
