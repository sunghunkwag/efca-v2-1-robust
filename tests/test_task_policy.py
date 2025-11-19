import unittest
import torch
from efca.policy.task_policy import TaskPolicy

class TestTaskPolicy(unittest.TestCase):
    """
    Unit tests for the TaskPolicy module.
    """

    def test_initialization(self):
        """
        Tests that the TaskPolicy module can be initialized without errors.
        """
        try:
            model = TaskPolicy(hidden_dim=64, action_dim=2)
            self.assertIsInstance(model, TaskPolicy)
        except Exception as e:
            self.fail(f"TaskPolicy initialization failed with an exception: {e}")

if __name__ == '__main__':
    unittest.main()
