import unittest
import torch
from efca.metacontrol.meta_controller import MetaController

class TestMetaController(unittest.TestCase):
    """
    Unit tests for the MetaController module.
    """

    def test_initialization(self):
        """
        Tests that the MetaController module can be initialized without errors.
        """
        try:
            model = MetaController(probe_dim=64)
            self.assertIsInstance(model, MetaController)
        except Exception as e:
            self.fail(f"MetaController initialization failed with an exception: {e}")

    def test_forward_pass_clamps_delta(self):
        """
        Tests that the forward pass of the MetaController clamps the output delta
        to the specified maximum value.
        """
        model = MetaController(probe_dim=64, max_delta=0.05)
        probe_output = torch.randn(1, 64)
        delta = model(probe_output)

        self.assertLessEqual(delta.abs().item(), 0.05 + 1e-9)

if __name__ == '__main__':
    unittest.main()
