import argparse
import importlib
import os
import pkgutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, Set, T, Union

from repro.commands.subcommand import RootSubcommand
from repro.common import Registrable

PathType = Union[os.PathLike, str]
ContextManagerFunctionReturnType = Generator[T, None, None]


@contextmanager
def push_python_path(path: PathType) -> ContextManagerFunctionReturnType[None]:
    """
    Prepends the given path to `sys.path`.
    This method is intended to use with `with`, so after its usage, its value willbe removed from
    `sys.path`.
    """
    # In some environments, such as TC, it fails when sys.path contains a relative path, such as ".".
    path = Path(path).resolve()
    path = str(path)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        # Better to remove by value, in case `sys.path` was manipulated in between.
        sys.path.remove(path)


# Borrowed from AllenNLP
def import_module_and_submodules(
    package_name: str, exclude: Optional[Set[str]] = None
) -> None:
    """
    Import all submodules under the given package.
    Primarily useful so that people using AllenNLP as a library
    can specify their own custom packages and have their custom
    classes get loaded and registered.
    """
    if exclude and package_name in exclude:
        return

    importlib.invalidate_caches()

    # For some reason, python doesn't always add this by default to your path, but you pretty much
    # always want it when using `--include-package`.  And if it's already there, adding it again at
    # the end won't hurt anything.
    with push_python_path("."):
        # Import at top level
        module = importlib.import_module(package_name)
        path = getattr(module, "__path__", [])
        path_string = "" if not path else path[0]

        # walk_packages only finds immediate children, so need to recurse.
        for module_finder, name, _ in pkgutil.walk_packages(path):
            # Sometimes when you import third-party libraries that are on your path,
            # `pkgutil.walk_packages` returns those too, so we need to skip them.
            if path_string and module_finder.path != path_string:
                continue
            subpackage = f"{package_name}.{name}"
            import_module_and_submodules(subpackage, exclude=exclude)


def build_argument_parser():
    # Ensure all of the subcommands have been loaded. We do not want to load
    # the _models directory because it has code that goes into the Dockerfile, not the
    # repro project. Each model's own import (under repro/models/<name>) will import only
    # the files that are necessary from _models
    import_module_and_submodules("repro", exclude={"repro.models._models"})

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Add all of the root-level commands using the registry
    for name, (cls_, _) in sorted(Registrable._registry[RootSubcommand].items()):
        cls_().add_subparser(subparsers)

    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    if "func" in dir(args):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
