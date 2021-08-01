import os
import pytest
from typing import List, Tuple

from repro.common import docker
from repro.common.util import NestedDict, flatten_nested_dict

FIXTURES_ROOT = f"{os.path.dirname(os.path.abspath(__file__))}/../tests/fixtures"


def get_testing_device_parameters(gpu_only: bool = False) -> List[Tuple[int]]:
    """
    Gets the list of device parameters to use for unittesting based
    on the value of the TEST_DEVICES environment variable. If `gpu_only`
    is `True`, the CPU will not be returned.
    """
    devices = os.environ.get("TEST_DEVICES", "-1").split(",")
    if gpu_only:
        if "-1" in devices:
            devices.remove("-1")
    return [(int(device),) for device in devices]


def assert_dicts_approx_equal(
    d1: NestedDict, d2: NestedDict, rel: float = None, abs: float = None
):
    d1 = flatten_nested_dict(d1)
    d2 = flatten_nested_dict(d2)
    assert d1.keys() == d2.keys()
    for key in d1.keys():
        assert d1[key] == pytest.approx(d2[key], rel=rel, abs=abs)


class DockerTestContextManager(object):
    def __init__(self, image: str):
        self.image = image

    def __enter__(self):
        # Make sure the image doesn't exist. If it does, delete it
        if docker.image_exists(self.image):
            docker.remove_image(self.image, force=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Make sure the image gets deleted
        if docker.image_exists(self.image):
            docker.remove_image(self.image, force=True)
