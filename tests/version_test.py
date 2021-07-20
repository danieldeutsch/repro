import unittest

import repro


class TestVersion(unittest.TestCase):
    def test_version_exists(self):
        assert repro.__version__
