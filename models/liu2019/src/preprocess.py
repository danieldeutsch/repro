import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import List

sys.path.append("PreSumm/src")

from others.utils import clean


def _tokenize(input_file: str) -> List[List[List[str]]]:
    """
    Tokenize the documents in the input file using the Stanford CoreNLP library.
    """
    with tempfile.TemporaryDirectory() as temp:
        # Serialize all of the data to disk, one document per file
        filenames = []
        input_dir = f"{temp}/input"
        os.makedirs(input_dir)
        with open(input_file, "r") as f:
            for i, line in enumerate(f):
                filename = f"{input_dir}/{i}"
                filenames.append(filename)
                with open(filename, "w") as out:
                    out.write(line)

        file_list = f"{temp}/mapping_for_corenlp.txt"
        with open(file_list, "w") as out:
            for filename in filenames:
                out.write(filename + "\n")

        # Run the tokenization
        output_dir = f"{temp}/output"
        os.makedirs(output_dir)
        print("Tokenizing documents with CoreNLP")
        command = [
            "java",
            "edu.stanford.nlp.pipeline.StanfordCoreNLP",
            "-annotators",
            "tokenize,ssplit",
            "-ssplit.newlineIsSentenceBreak",
            "always",
            "-filelist",
            file_list,
            "-outputFormat",
            "json",
            "-outputDirectory",
            output_dir,
        ]
        subprocess.call(command)
        print("Finished tokenizing documents")

        # Load the results
        documents = []
        for i in range(len(filenames)):
            results = json.load(open(f"{output_dir}/{i}.json"))
            sentences = []
            for sentence in results["sentences"]:
                tokens = [token["word"].lower() for token in sentence["tokens"]]
                sentences.append(tokens)

            sentences = [clean(" ".join(sentence)).split() for sentence in sentences]
            documents.append(sentences)

        return documents


def main(args):
    documents = _tokenize(args.input_file)

    dirname = os.path.dirname(args.output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(args.output_file, "w") as out:
        for document in documents:
            document = " [CLS] [SEP] ".join(
                [" ".join(sentence) for sentence in document]
            )
            out.write(document + "\n")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
