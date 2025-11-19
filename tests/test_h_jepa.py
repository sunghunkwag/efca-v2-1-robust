import unittest
import torch
from efca.perception.h_jepa import HJEPA

class TestHJEPA(unittest.TestCase):
    """
    Unit tests for the H-JEPA module.
    """

    def test_initialization(self):
        """
        Tests that the H-JEPA module can be initialized without errors.
        """
        try:
            model = HJEPA()
            self.assertIsInstance(model, HJEPA)
        except Exception as e:
            self.fail(f"HJEPA initialization failed with an exception: {e}")

    def test_forward_pass(self):
        """
        Tests the forward pass of the H-JEPA module.
        """
        try:
            model = HJEPA()
            dummy_input = torch.randn(1, 3, 224, 224)
            loss, features = model(dummy_input)
            self.assertIsInstance(loss, torch.Tensor)
            self.assertIsInstance(features, torch.Tensor)
        except Exception as e:
            self.fail(f"HJEPA forward pass failed with an exception: {e}")

if __name__ == '__main__':
    unittest.main()
