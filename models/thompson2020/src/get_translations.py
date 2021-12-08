import argparse


def main(args):
    id_to_translation = {}
    prefix = f"<{args.language}> "

    with open(args.input_file, "r") as f:
        for line in f:
            if line.startswith("H-"):
                columns = line.strip().split("\t")
                id_ = int(columns[0][2:])
                translation_with_language = columns[2]
                if not translation_with_language.startswith(prefix):
                    raise Exception(
                        f'Expected language prefix "{prefix}" on the translation: {translation_with_language}'
                    )
                translation = translation_with_language[len(prefix) :]
                id_to_translation[id_] = translation

    with open(args.output_file, "w") as out:
        for id_ in range(len(id_to_translation)):
            out.write(id_to_translation[id_] + "\n")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--language", required=True)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
