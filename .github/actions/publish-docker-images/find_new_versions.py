import argparse


def main(args):
    pass


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--master-sha", required=True)
    argp.add_argument("--current-sha", required=True)
    args = argp.parse_args()
    main(args)
