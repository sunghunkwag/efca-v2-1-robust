import unittest
import torch
from efca.dynamics.ct_lnn import CTLNN

class TestCTLNN(unittest.TestCase):
    """
    Unit tests for the CT-LNN module.
    """

    def test_initialization(self):
        """
        Tests that the CT-LNN module can be initialized without errors.
        """
        try:
            model = CTLNN(input_dim=128, hidden_dim=64, output_dim=32)
            self.assertIsInstance(model, CTLNN)
        except Exception as e:
            self.fail(f"CTLNN initialization failed with an exception: {e}")

if __name__ == '__main__':
    unittest.main()
