"""
This script was adapted from https://github.com/tagoyal/dae-factuality/blob/d5a472b0f18c18d7125dc1551bde59538adac765/evaluate_factuality.py
to calculate the scores for input text rather than evaluate the accuracy.
"""
# fmt: off
import argparse
import os
import json
import numpy as np
import torch
import utils
from sklearn.utils.extmath import softmax
from preprocessing_utils import get_tokens, get_relevant_deps_and_context
from transformers import (
    BertConfig,
    BertTokenizer,
    ElectraConfig,
    ElectraTokenizer,
)

MODEL_CLASSES = {"bert_dae": (BertConfig, utils.BertDAEModel, BertTokenizer),
                 "bert_basic": (BertConfig, utils.BERTBasicModel, BertTokenizer),
                 "electra_basic": (ElectraConfig, utils.ElectraBasicModel, ElectraTokenizer),
                 "electra_dae": (ElectraConfig, utils.ElectraDAEModel, ElectraTokenizer), }


def score_example_single_context(decode_text, input_text, model, tokenizer, args):
    gen_tok, _, gen_dep = get_relevant_deps_and_context(decode_text, args)

    tokenized_text = get_tokens(input_text)

    ex = {'input': tokenized_text, 'deps': [], 'context': ' '.join(gen_tok), 'sentlabel': -1}
    for dep in gen_dep:
        ex['deps'].append({'dep': dep['dep'], 'label': -1, 'head_idx': dep['head_idx'] - 1,
                           'child_idx': dep['child_idx'] - 1, 'child': dep['child'], 'head': dep['head']})

    dict_temp = {'id': 0, 'input': ex['input'], 'sentlabel': ex['sentlabel'], 'context': ex['context']}
    for i in range(20):
        if i < len(ex['deps']):
            dep = ex['deps'][i]
            dict_temp['dep_idx' + str(i)] = str(dep['child_idx']) + ' ' + str(dep['head_idx'])
            dict_temp['dep_words' + str(i)] = str(dep['child']) + ' ' + str(dep['head'])
            dict_temp['dep' + str(i)] = dep['dep']
            dict_temp['dep_label' + str(i)] = dep['label']
        else:
            dict_temp['dep_idx' + str(i)] = ''
            dict_temp['dep_words' + str(i)] = ''
            dict_temp['dep' + str(i)] = ''
            dict_temp['dep_label' + str(i)] = ''

    features = utils.convert_examples_to_features_bert(
        [dict_temp],
        tokenizer,
        max_length=128,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
    )

    # Convert to Tensors and build dataset
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(args.device)
    attention = torch.tensor([f.input_attention_mask for f in features], dtype=torch.long).to(args.device)
    token_ids = torch.tensor([f.token_ids for f in features], dtype=torch.long).to(args.device)

    child = torch.tensor([f.child_indices for f in features], dtype=torch.long).to(args.device)
    head = torch.tensor([f.head_indices for f in features], dtype=torch.long).to(args.device)

    dep_labels = torch.tensor([f.dep_labels for f in features], dtype=torch.long).to(args.device)
    num_dependencies = torch.tensor([f.num_dependencies for f in features], dtype=torch.long).to(args.device)
    arcs = torch.tensor([f.arcs for f in features], dtype=torch.long).to(args.device)
    arc_labels = torch.tensor([f.arc_labels for f in features], dtype=torch.long).to(args.device)
    arc_label_lengths = torch.tensor([f.arc_label_lengths for f in features], dtype=torch.long).to(args.device)

    inputs = {'input_ids': input_ids, 'attention': attention, 'token_ids': token_ids, 'child': child, 'head': head,
              'dep_labels': dep_labels, 'arcs': arc_labels, 'arc_label_lengths': arc_label_lengths,
              'device': args.device}

    outputs = model(**inputs)
    tmp_eval_loss, logits = outputs[:2]
    preds = logits.detach().cpu().numpy()

    f_out = open('test.txt', 'a')
    text = tokenizer.decode(input_ids[0])
    text = text.replace(tokenizer.pad_token, '').strip()
    f_out.write(text + '\n')
    for j, arc in enumerate(arcs[0]):
        arc_text = tokenizer.decode(arc)
        arc_text = arc_text.replace(tokenizer.pad_token, '').strip()

        if arc_text == '':
            break

        pred_temp = softmax([preds[0][j]])
        f_out.write(arc_text + '\n')
        f_out.write('pred:\t' + str(np.argmax(pred_temp)) + '\n')
        f_out.write(str(pred_temp[0][0]) + '\t' + str(pred_temp[0][1]) + '\n')
        f_out.write('\n')

    f_out.close()
    preds = preds.reshape(-1, 2)
    preds = softmax(preds)
    preds = preds[:, 1]
    preds = preds[:num_dependencies[0]]
    score = np.mean(preds)
    return score


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--include_sentence_level", action="store_true", help="Overwrite the cached data sets", )
    parser.add_argument("--gpu_device", type=int, default=0, help="gpu device")
    parser.add_argument("--dependency_type", default='enhancedDependencies', help='type of dependency labels')

    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)

    args = parser.parse_args()

    args.n_gpu = 1
    if args.gpu_device != -1:
        device = torch.device("cuda", args.gpu_device)
    else:
        device = torch.device("cpu")
    args.device = device

    print(args.input_dir)
    if not os.path.exists(args.input_dir):
        print('Check model location')

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(args.input_dir)
    model = model_class.from_pretrained(args.input_dir)
    model.to(args.device)

    with open(args.output_file, "w") as out:
        with open(args.input_file, "r") as f:
            for line in f:
                data = json.loads(line)
                candidate = data["candidate"]
                source = data["source"]

                score = score_example_single_context(candidate, source, model, tokenizer, args)
                out.write(json.dumps({"dae": float(score)}) + "\n")


if __name__ == "__main__":
    main()
