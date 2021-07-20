import os
from typing import List, Tuple


def get_testing_device_parameters() -> List[Tuple[int]]:
    """
    Gets the list of device parameters to use for unittesting based
    on the value of the TEST_DEVICES environment variable.
    """
    devices = os.environ.get("TEST_DEVICES", "-1").split(",")
    return [(int(device),) for device in devices]
