import unittest
import torch
from efca.probe.probe_network import ProbeNetwork

class TestProbeNetwork(unittest.TestCase):
    """
    Unit tests for the ProbeNetwork module.
    """

    def test_initialization(self):
        """
        Tests that the ProbeNetwork module can be initialized without errors.
        """
        try:
            model = ProbeNetwork(hidden_dim=256, probe_dim=64)
            self.assertIsInstance(model, ProbeNetwork)
        except Exception as e:
            self.fail(f"ProbeNetwork initialization failed with an exception: {e}")

    def test_forward_pass_detaches_gradient(self):
        """
        Tests that the forward pass of the ProbeNetwork detaches the hidden state
        from the computation graph, preventing gradients from flowing back.
        """
        model = ProbeNetwork(hidden_dim=256, probe_dim=64)
        h = torch.randn(1, 256, requires_grad=True)

        # We need to use a hook to inspect the gradient of the input to the first layer
        grad_fn_checker = {}
        def hook(module, input, output):
            grad_fn_checker['grad_fn'] = input[0].grad_fn

        model.probe_net[0].register_forward_hook(hook)

        model(h)

        self.assertIsNone(grad_fn_checker['grad_fn'])

if __name__ == '__main__':
    unittest.main()
