# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = "Repro"
copyright = "2022, Daniel Deutsch"
author = "Daniel Deutsch"

# version.py defines the VERSION variable.
# We use exec here so we don't import repro whilst setting up.
VERSION = {}
with open("../../repro/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)
release = VERSION["VERSION"]


# -- General configuration ---------------------------------------------------
display_version = False

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = []

extensions.append("sphinx.ext.napoleon")
napoleon_google_docstring = False

extensions.append("sphinx.ext.autodoc")
autodoc_typehints = "none"

extensions.append("myst_parser")
myst_heading_anchors = 3

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_title = f"Repro v{release}"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# When we build the docs locally, we do so from the "docs" directory, but
# readthedocs does from the "docs/source" directory, which messes up the
# relative directory names that we use to automatically generate the
# documentation. This hack checks to see what directory we are in so
# we can adjust the paths accordingly
print("CWD", os.getcwd())
is_readthedocs_build = os.getcwd().endswith("docs/source")


def generate_apidocs():
    """
    Walk the source code and create rst files for each module. We don't use
    sphinx-apidoc because we want additional customization that was difficult
    (or not possible) to do with sphinx-apidoc.
    """
    import pkgutil
    import shutil
    from typing import List

    target_dir = "api" if is_readthedocs_build else "source/api"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    exclude = {
        "repro.__main__",
        "repro.version",
    }

    def _process_module(name: str, prefix: List[str]):
        qualified_name = ".".join(prefix + [name])
        if qualified_name in exclude:
            return

        with open(f"{target_dir}/{qualified_name}.md", "w") as out:
            out.write(f"# {qualified_name}\n")
            out.write(f"```{{eval-rst}}\n")
            out.write(f".. automodule:: {qualified_name}\n")
            out.write(f"    :members:\n")
            out.write(f"    :undoc-members:\n")
            out.write(f"```\n")

    def _process_package(submodules: List[str], prefix: List[str]):
        package = ".".join(prefix)

        with open(f"{target_dir}/{package}.md", "w") as out:
            out.write(f"# {package}\n")
            out.write(f"```{{eval-rst}}\n")
            out.write(f".. toctree::\n")
            out.write(f"    :hidden:\n")
            out.write(f"\n")
            for module in sorted(submodules):
                module_name = f"{package}.{module}"
                if module_name in exclude:
                    continue

                if len(prefix) >= 2:
                    display_name = module
                else:
                    display_name = module_name
                out.write(f"    {display_name}<{module_name}>\n")

            out.write(f"```\n")

    def _generate(path: str, prefix: List[str]):
        qualified_name = ".".join(prefix)

        children = []
        for _, name, is_package in pkgutil.iter_modules([path]):
            # We don't want to do anything for subpackages
            # of repro.models
            if qualified_name == "repro.models" and is_package:
                continue

            children.append(name)
            if is_package:
                # Recurse
                _generate(f"{path}/{name}", prefix + [name])
            else:
                # Generate a file for the module
                _process_module(name, prefix)

        # Generate a file for this package
        _process_package(children, prefix)

    starting_path = "../../../repro/repro" if is_readthedocs_build else "../../repro/repro"
    _generate(starting_path, ["repro"])


def generate_model_files():
    """
    This function copies over all of the model's individual Readmes
    into the `source/models` directory and creates `source/models/index.md`.
    """
    print(f"Generating model files from {os.getcwd()}")

    import shutil
    from glob import glob

    target_dir = "models"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    model_dir = "../../../repro/models" if is_readthedocs_build else "../../repro/models"
    models = []
    for readme_path in glob(f"{model_dir}/*/Readme.md"):
        print("Processing " + readme_path)
        model = os.path.basename(os.path.dirname(readme_path))
        dst = f"{target_dir}/{model}.md"
        shutil.copyfile(readme_path, dst)
        models.append(model)
    models.sort()

    with open(f"{target_dir}/index.md", "w") as out:
        out.write("# Models\n")
        out.write("```{eval-rst}\n")
        out.write(".. toctree::\n")
        out.write("\t:hidden:\n")
        out.write("\n")
        for model in models:
            out.write(f"\t{model}\n")
        out.write("```\n")


def setup(app):
    generate_apidocs()
    generate_model_files()
