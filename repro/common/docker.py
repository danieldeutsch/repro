import argparse
import docker
import json
import logging
from typing import Dict, Optional

from overrides import overrides

from repro.commands.subcommand import SetupSubcommand
from repro.common import REPRO_CONFIG
from repro.common.logging import prepare_global_logging

logger = logging.getLogger(__name__)


def make_volume_map(*volumes: str) -> Dict[str, str]:
    volumes_dict = {}
    for i, volume in enumerate(volumes):
        volumes_dict[volume] = f"/tmp{i}"
    return volumes_dict


def image_exists(image: str) -> bool:
    client = docker.from_env()
    try:
        client.images.get(image)
    except docker.errors.ImageNotFound:
        return False
    return True


def remove_image(image: str, force: bool = False) -> None:
    client = docker.from_env()
    client.images.remove(image, force)


def run_command(
    image: str,
    command: str,
    volume_map: Optional[Dict[str, str]] = None,
    stdout: bool = True,
    stderr: bool = True,
    silent: bool = False,
    network_disabled: bool = False,
    cuda: bool = False,
) -> str:
    """
    Runs a shell command in a Docker image.

    Parameters
    ----------
    image : str
        The name of the Docker image
    command : str
        The command to run
    volume_map : Dict[str, str], default=None
        A mapping between host directories to container directories
    stdout : bool, default=True
        Indicates the stdout stream should be returned
    stderr : bool, default=True
        Indicates the stderr stream should be returned
    silent : bool, default=False
        Indicates whether the stdout and stderr streams of the command should be written to stdout
        if `stdout` and `stderr` are `True`, respectively
    network_disabled : bool, default=False
        Indicates the container's network connection should be disabled
    cuda : bool, default=False
        Indicates that the processes uses cuda, in which the runtime will be set to "nvidia"

    Returns
    -------
    str
        The stdout and stderr streams of the command if `stdout` and `stderr` are `True`, respectively.
        There appears to be a bug in the `docker` package related to collecting the full stdout and stderr streams, so
        we do not recommend relying on these streams.
    """
    if "'" in command:
        raise ValueError(
            f'The command contains the character "\'", which is currently not supported: {command}'
        )
    if not stdout and not stderr:
        raise ValueError(
            f"The `docker` package requires either `stdout` or `stderr` is `True`"
        )

    volume_map = volume_map or {}
    volumes = {
        host_path: {"bind": container_path, "mode": "rw"}
        for host_path, container_path in volume_map.items()
    }

    runtime = "nvidia" if cuda else None

    docker_command = f"/bin/sh -c '{command}'"
    logger.info(f'Running command in Docker image {image}: "{docker_command}"')

    client = docker.from_env()
    container = client.containers.run(
        image,
        docker_command,
        volumes=volumes,
        detach=True,
        network_disabled=network_disabled,
        runtime=runtime,
    )
    logs = container.logs(stream=True, stdout=stdout, stderr=stderr)

    # Collect the stdout and/or stderr, printing if verbose
    output = []
    for item in logs:
        item = item.decode()
        output.append(item)
        if not silent:
            print(item, end="")
    logger.info("Command finished")

    output = "".join(output)
    return output


def build_image(
    root: str, image: str, build_args: Dict[str, str] = None, silent: bool = False
) -> None:
    logger.info(f"Building image {image} with Dockerfile in directory {root}")
    docker_server = REPRO_CONFIG["docker_server"]
    client = docker.APIClient(base_url=docker_server)
    generator = client.build(path=root, tag=image, buildargs=build_args, rm=True)
    try:
        for response in generator:
            response = response.decode()
            for json_obj in response.split("\r\n"):
                if json_obj:
                    data = json.loads(json_obj)
                    if "stream" in data:
                        stream = data["stream"]
                        if not silent:
                            print(stream, end="")
    except Exception as e:
        logger.error("Error building image")
        logger.exception(e)

    logger.info("Finished building image")


class DockerContextManager(object):
    def __init__(self, image: str):
        self.image = image

    def __enter__(self):
        # Make sure the image doesn't exist. If it does, delete it
        if image_exists(self.image):
            remove_image(self.image, force=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Make sure the image gets deleted
        if image_exists(self.image):
            remove_image(self.image, force=True)


class BuildDockerImageSubcommand(SetupSubcommand):
    def __init__(
        self,
        image: str,
        root: str,
    ) -> None:
        self.image = image
        self.root = root

    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction):
        description = f'Build the docker image "{self.image}"'
        self.parser = parser.add_parser(
            self.image, description=description, help=description
        )
        self.parser.add_argument(
            "--silent",
            action="store_true",
            help="Silences the output from the build command",
        )
        self.parser.set_defaults(subfunc=self.run)

    @overrides
    def run(self, args):
        prepare_global_logging(silent=args.silent)
        build_image(self.root, self.image, silent=args.silent)
