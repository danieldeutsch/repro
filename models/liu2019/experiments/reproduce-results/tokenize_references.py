"""
Tokenizes the reference summaries using the instructions from Liu & Lapata (2019)
"""
import argparse
import json
import os

from repro.common import TemporaryDirectory
from repro.common.docker import make_volume_map, run_command
from repro.common.io import write_to_text_file, read_jsonl_file
from repro.common.util import flatten


def main(args):
    instances = read_jsonl_file(args.input_jsonl)
    references = [flatten(instance["reference"]["text"]) for instance in instances]

    with TemporaryDirectory() as temp:
        host_input_dir = f"{temp}/input"
        host_output_dir = f"{temp}/output"
        volume_map = make_volume_map(host_input_dir, host_output_dir)
        container_input_dir = volume_map[host_input_dir]
        container_output_dir = volume_map[host_output_dir]

        host_input_file = f"{host_input_dir}/references.txt"
        container_input_file = f"{container_input_dir}/references.txt"

        write_to_text_file(references, host_input_file)

        # Run inference. The output_dir must exist before running
        # the docker command
        os.makedirs(host_output_dir)
        host_output_file = f"{host_output_dir}/references.txt"
        container_output_file = f"{container_output_dir}/references.txt"
        command = (
            f"python preprocess.py"
            f"  --input-file {container_input_file}"
            f"  --output-file {container_output_file}"
        )

        run_command(
            "liu2019",
            command,
            volume_map=volume_map,
            network_disabled=True,
        )

        # Load the output summaries
        references = []
        with open(host_output_file, "r") as f:
            for line in f:
                sentences = line.strip().split(" [CLS] [SEP] ")
                references.append(sentences)

    with open(args.output_jsonl, "w") as out:
        for instance, reference in zip(instances, references):
            instance["reference"] = {"text": reference}
            out.write(json.dumps(instance) + "\n")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-jsonl", required=True)
    argp.add_argument("--output-jsonl", required=True)
    args = argp.parse_args()
    main(args)
