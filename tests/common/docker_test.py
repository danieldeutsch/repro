import json
import unittest

from repro.common import TemporaryDirectory, docker
from repro.testing import FIXTURES_ROOT


class TestDocker(unittest.TestCase):
    def test_make_volume_map(self):
        volume_map = docker.make_volume_map()
        assert volume_map == {}

        volume_map = docker.make_volume_map("vol1")
        assert volume_map == {"vol1": "/tmp0"}

        volume_map = docker.make_volume_map("vol1", "vol2")
        assert volume_map == {"vol1": "/tmp0", "vol2": "/tmp1"}

    def test_image_exists_build_remove(self):
        # First remove the image if it exists
        image = "python-3.8"
        if docker.image_exists(image):
            docker.remove_image(image)

        assert not docker.image_exists(image)
        docker.build_image(f"{FIXTURES_ROOT}/dockerfiles/python-3.8", image)
        assert docker.image_exists(image)

        docker.remove_image(image)
        assert not docker.image_exists(image)

    def test_build_image_build_args(self):
        """
        Tests whether `build_image` correctly passes build arguments (roughly equivalent
        to command line arguments). The docker image creates environment variables for each
        of the build arguments here and saves a file that says what the environment variables
        were set to.
        """
        image = "build-image"

        # None are set
        with docker.DockerContextManager(image):
            with TemporaryDirectory() as temp:
                assert not docker.image_exists(image)
                docker.build_image(
                    f"{FIXTURES_ROOT}/dockerfiles/build-image", image, silent=True
                )

                volume_map = docker.make_volume_map(temp)
                output_dir = volume_map[temp]
                output_file = f"{temp}/results.json"

                docker.run_command(
                    image,
                    f"python run.py {output_dir}",
                    volume_map=volume_map,
                    silent=True,
                )
                results = json.load(open(output_file, "r"))
                assert results["ENV1"] == ""
                assert results["ENV2"] == ""

        # One is set
        with docker.DockerContextManager(image):
            with TemporaryDirectory() as temp:
                assert not docker.image_exists(image)
                docker.build_image(
                    f"{FIXTURES_ROOT}/dockerfiles/build-image",
                    image,
                    build_args={"ENV1": "set"},
                    silent=True,
                )

                volume_map = docker.make_volume_map(temp)
                output_dir = volume_map[temp]
                output_file = f"{temp}/results.json"

                docker.run_command(
                    image,
                    f"python run.py {output_dir}",
                    volume_map=volume_map,
                    silent=True,
                )
                results = json.load(open(output_file, "r"))
                assert results["ENV1"] == "set"
                assert results["ENV2"] == ""

        # Both are set
        with docker.DockerContextManager(image):
            with TemporaryDirectory() as temp:
                assert not docker.image_exists(image)
                docker.build_image(
                    f"{FIXTURES_ROOT}/dockerfiles/build-image",
                    image,
                    build_args={"ENV1": "set", "ENV2": "also set"},
                    silent=True,
                )

                volume_map = docker.make_volume_map(temp)
                output_dir = volume_map[temp]
                output_file = f"{temp}/results.json"

                docker.run_command(
                    image,
                    f"python run.py {output_dir}",
                    volume_map=volume_map,
                    silent=True,
                )
                results = json.load(open(output_file, "r"))
                assert results["ENV1"] == "set"
                assert results["ENV2"] == "also set"

    def test_run_command_volume_map(self):
        """
        Tests whether `run_command` successfully mounts host directories to a container directory.
        The docker image writes a file which contains whether or not the directories passed
        as arguments to the python script exist or not.
        """
        image = "build-image"

        with docker.DockerContextManager(image):
            with TemporaryDirectory() as temp:
                assert not docker.image_exists(image)
                docker.build_image(
                    f"{FIXTURES_ROOT}/dockerfiles/build-image", image, silent=True
                )

                host_dir1 = temp
                host_dir2 = f"{temp}/dir2"
                host_dir3 = f"{temp}/dir3"
                volume_map = docker.make_volume_map(host_dir1, host_dir2, host_dir3)
                container_dir1 = volume_map[host_dir1]
                container_dir2 = volume_map[host_dir2]
                container_dir3 = volume_map[host_dir3]

                output_file = f"{temp}/results.json"
                docker.run_command(
                    image,
                    f"python run.py {container_dir1} {container_dir2} {container_dir3}",
                    volume_map={host_dir1: container_dir1},
                    silent=True,
                )
                results = json.load(open(output_file, "r"))

                assert results[container_dir1] is True
                assert results[container_dir2] is False
                assert results[container_dir3] is False

                output_file = f"{temp}/results.json"
                docker.run_command(
                    image,
                    f"python run.py {container_dir1} {container_dir2} {container_dir3}",
                    volume_map={host_dir1: container_dir1, host_dir2: container_dir2},
                    silent=True,
                )
                results = json.load(open(output_file, "r"))

                assert results[container_dir1] is True
                assert results[container_dir2] is True
                assert results[container_dir3] is False

                output_file = f"{temp}/results.json"
                docker.run_command(
                    image,
                    f"python run.py {container_dir1} {container_dir2} {container_dir3}",
                    volume_map={
                        host_dir1: container_dir1,
                        host_dir2: container_dir2,
                        host_dir3: container_dir3,
                    },
                    silent=True,
                )
                results = json.load(open(output_file, "r"))

                assert results[container_dir1] is True
                assert results[container_dir2] is True
                assert results[container_dir3] is True

    def test_network_disabled(self):
        """
        Tests to ensure that the container's network is properly disabled. The
        docker container pings google and writes whether or not it was successful to
        an output file.
        """
        image = "network-disabled"
        with docker.DockerContextManager(image):
            with TemporaryDirectory() as temp:
                volume_map = docker.make_volume_map(temp)

                docker.build_image(
                    f"{FIXTURES_ROOT}/dockerfiles/network-disabled", image
                )
                docker.run_command(image, "sh run.sh", volume_map=volume_map)
                assert open(f"{temp}/results.txt").read().strip() == "Online"

                docker.run_command(
                    image, "sh run.sh", volume_map=volume_map, network_disabled=True
                )
                assert open(f"{temp}/results.txt").read().strip() == "Offline"


class TestDockerContextManager(unittest.TestCase):
    def test_docker_context_manager(self):
        # Create an image, ensure the context manager deletes it,
        # recreate it, then make sure it gets deleted on the context manager exit
        image = "python-3.8"
        if not docker.image_exists(image):
            docker.build_image(
                f"{FIXTURES_ROOT}/dockerfiles/python-3.8", image, silent=True
            )

        assert docker.image_exists(image)
        with docker.DockerContextManager(image):
            assert not docker.image_exists(image)
            docker.build_image(
                f"{FIXTURES_ROOT}/dockerfiles/python-3.8", image, silent=True
            )
            assert docker.image_exists(image)
        assert not docker.image_exists(image)
