from typing import List

from repro.common import Registrable
from repro.data.types import InstanceDict


class DatasetReader(Registrable):
    def read(self, *input_files: str) -> List[InstanceDict]:
        """
        Loads the instances from the `input_files` by calling the `_read` method.
        After the instances have been loaded, this method verifies that the
        instances all have an "instance_id" key.

        Parameters
        ----------
        input_files : str
            The input files

        Returns
        -------
        List[InstanceDict]
            The instances
        """
        instances = self._read(*input_files)

        # Ensure all instances at least have an instance_id
        for instance in instances:
            if "instance_id" not in instance:
                raise Exception(
                    "All instances returned from a `DatasetReader` are required "
                    "to have an `instance_id` key."
                )
            else:
                instance_id = instance["instance_id"]
                if not isinstance(instance_id, str):
                    raise Exception(
                        f"The value for key `instance_id` is required to be a string. Found: {type(instance_id)}"
                    )

        return instances

    def _read(self, *input_files: str) -> List[InstanceDict]:
        """
        Loads the instances from the `input_files`. Each of the instances should
        have an "instance_id" key with a value that uniquely identifies that instance.

        Parameters
        ----------
        input_files : str
            The input files

        Returns
        -------
        List[InstanceDict]
            The instances
        """
        raise NotImplementedError
