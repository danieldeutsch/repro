import argparse
import importlib
import pkgutil


def main(args):
    versions = []
    for module_info in pkgutil.iter_modules(["repro/models"]):
        name = module_info.name
        module_name = f"repro.models.{name}"
        module = importlib.import_module(module_name)
        try:
            version = module.VERSION
            repository = module.DOCKERHUB_REPRO
            versions.append((name, repository, version))
        except AttributeError:
            print(f"{module_name} does not have a version and/or DockerHub repository")

    with open(args.output_file, "w") as out:
        for model, repository, version in versions:
            out.write(f"{model} {repository} {version}\n")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
