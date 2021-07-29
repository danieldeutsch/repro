import os
from typing import List, Tuple


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
