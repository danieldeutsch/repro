import argparse
import importlib
import pkgutil
from overrides import overrides

from repro.commands.subcommand import RootSubcommand
from repro.common.docker import pull_image
from repro.common.logging import prepare_global_logging


@RootSubcommand.register("pull")
class PullImageSubcommand(RootSubcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction):
        description = "Pull a model's default Docker image from Docker Hub"
        self.parser = parser.add_parser(
            "pull", description=description, help=description
        )

        # Find all of the modules that have default Docker Hub repos
        self.images = {}
        for module_info in pkgutil.iter_modules(["repro/models"]):
            module_name = f"repro.models.{module_info.name}"
            module = importlib.import_module(module_name)

            try:
                model = module.MODEL_NAME
                image = module.DEFAULT_IMAGE
                self.images[model] = image
            except AttributeError:
                # We can't find a default image
                pass

        choices = sorted(self.images.keys())
        self.parser.add_argument(
            "model",
            help="The name of the model to pull the Docker image for",
            choices=choices,
        )

        self.parser.set_defaults(func=self.run)

    @overrides
    def run(self, args):
        prepare_global_logging()
        image = self.images[args.model]
        pull_image(image)
