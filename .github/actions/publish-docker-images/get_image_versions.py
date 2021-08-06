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
        except AttributeError:
            print(f"{module_name} does not have `VERSION` and/or `DOCKERHUB_REPRO` attributes. Will not publish")
            continue

        try:
            automatically_publish = module.AUTOMATICALLY_PUBLISH
        except AttributeError:
            print(f"{module_name} does not have an `AUTOMATICALLY_PUBLISH` attribute. Will not publish")
            continue

        if automatically_publish:
            versions.append((name, repository, version))
        else:
            print(f"{module_name} is set to not be automatically published. Will not publish")

    with open(args.output_file, "w") as out:
        for model, repository, version in versions:
            out.write(f"{model} {repository} {version}\n")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
