import argparse
import docker
import json
import logging
import shutil
import tempfile
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
    try:
        client = docker.from_env()
    except docker.errors.DockerException as e:
        logger.error("Could not connect to the Docker client. Is the Daemon running?")
        logger.exception(e)
        return False
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

    # Escape any single quotes from the command
    command = command.replace("'", "'\\''")

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


def pull_image(image: str) -> None:
    """
    Pulls an image from Docker Hub.

    Parameters
    ----------
    image : str
        The name of the image to pull.
    """
    logger.info(f"Pulling image {image} from Docker Hub")
    client = docker.from_env()
    client.images.pull(image)
    logger.info("Pulled image")


class BuildDockerImageSubcommand(SetupSubcommand):
    def __init__(self, root: str, default_image: str) -> None:
        self.root = root
        self.default_image = default_image

    @overrides
    def add_subparser(self, model: str, parser: argparse._SubParsersAction):
        description = f'Build a Docker image for model "{model}"'
        self.parser = parser.add_parser(
            model, description=description, help=description
        )
        self.parser.add_argument(
            "--image-name",
            default=self.default_image,
            help="The name of the image to build",
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
        build_image(self.root, args.image_name, silent=args.silent)


class DockerContainer(object):
    def __init__(self, image: str):
        self.image = image

    def __enter__(self):
        self.host_dir = tempfile.mkdtemp()
        self.volume_map = make_volume_map(self.host_dir)
        self.container_dir = self.volume_map[self.host_dir]
        return self

    def __exit__(self, *args):
        shutil.rmtree(self.host_dir)

    def run_command(self, **kwargs) -> str:
        for arg in ["image", "volume_map"]:
            if arg in kwargs:
                raise Exception(
                    f"`{arg}` parameter cannot be passed to the `DockerContainer`"
                    f"`run_command` function."
                )
        return run_command(self.image, volume_map=self.volume_map, **kwargs)
