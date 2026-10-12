"""Check add(2, 3) from outside the writable work tree."""

import unittest

from adder import add


class AdderTest(unittest.TestCase):
    """The operator's check. The model cannot replace this file."""

    def test_two_plus_three(self) -> None:
        """add(2, 3) is 5."""
        self.assertEqual(add(2, 3), 5)


if __name__ == "__main__":
    unittest.main()
