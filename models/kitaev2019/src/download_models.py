import argparse
import benepar


def main(args):
    for model in args.models.split(","):
        benepar.download(model)


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--models", required=True)
    args = argp.parse_args()
    main(args)
