
import unittest
import pytest
from ..src import tools


class TestTools(unittest.TestCase):

    def test_get42(self):
        should_be_42 = tools.get42()
        self.assertEqual(should_be_42, 42)

        

