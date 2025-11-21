import unittest
import torch
from efca.probe.probe_network import ProbeNetwork


class TestProbeNetwork(unittest.TestCase):
    """
    Unit tests for the Probe Network module.
    """

    def setUp(self):
        """Set up test configuration."""
        self.config = {
            'h_jepa_dim': 768,
            'gwt_dim': 768,
            'lnn_dim': 256,
            'output_dim': 64,
            'hidden_dim': 128
        }
        self.batch_size = 2

    def test_initialization(self):
        """Tests that the Probe Network can be initialized without errors."""
        try:
            probe = ProbeNetwork(self.config)
            self.assertIsInstance(probe, ProbeNetwork)
            self.assertEqual(probe.output_dim, 64)
            self.assertEqual(probe.hidden_dim, 128)
        except Exception as e:
            self.fail(f"ProbeNetwork initialization failed with an exception: {e}")

    def test_probe_h_jepa(self):
        """Tests the H-JEPA probe method."""
        probe = ProbeNetwork(self.config)

        # Create dummy H-JEPA features (B, C, H, W)
        features = torch.randn(self.batch_size, 768, 7, 7)
        loss = torch.tensor(0.5)

        output = probe.probe_h_jepa(features, loss)

        self.assertEqual(output.shape, (self.batch_size, self.config['hidden_dim'] // 2))
        self.assertIsInstance(output, torch.Tensor)

    def test_probe_gwt(self):
        """Tests the GWT probe method."""
        probe = ProbeNetwork(self.config)

        # Create dummy GWT slots (B, Num_Slots, Dim)
        slots = torch.randn(self.batch_size, 4, 768)

        output = probe.probe_gwt(slots)

        self.assertEqual(output.shape, (self.batch_size, self.config['hidden_dim'] // 2))
        self.assertIsInstance(output, torch.Tensor)

    def test_probe_lnn(self):
        """Tests the CT-LNN probe method."""
        probe = ProbeNetwork(self.config)

        # Create dummy CT-LNN hidden state (B, D)
        hidden_state = torch.randn(self.batch_size, 256)

        output = probe.probe_lnn(hidden_state)

        self.assertEqual(output.shape, (self.batch_size, self.config['hidden_dim'] // 2))
        self.assertIsInstance(output, torch.Tensor)

    def test_forward_pass(self):
        """Tests the full forward pass of the Probe Network."""
        probe = ProbeNetwork(self.config)

        # Create dummy inputs
        h_jepa_features = torch.randn(self.batch_size, 768, 7, 7)
        gwt_slots = torch.randn(self.batch_size, 4, 768)
        lnn_state = torch.randn(self.batch_size, 256)
        h_jepa_loss = torch.tensor(0.5)

        output = probe(h_jepa_features, gwt_slots, lnn_state, h_jepa_loss)

        self.assertEqual(output.shape, (self.batch_size, self.config['output_dim']))
        self.assertIsInstance(output, torch.Tensor)

    def test_get_statistics(self):
        """Tests the statistics extraction method."""
        probe = ProbeNetwork(self.config)

        # Create dummy inputs
        h_jepa_features = torch.randn(self.batch_size, 768, 7, 7)
        gwt_slots = torch.randn(self.batch_size, 4, 768)
        lnn_state = torch.randn(self.batch_size, 256)

        stats = probe.get_statistics(h_jepa_features, gwt_slots, lnn_state)

        self.assertIsInstance(stats, dict)
        self.assertIn('h_jepa_mean', stats)
        self.assertIn('gwt_std', stats)
        self.assertIn('lnn_max', stats)
        self.assertEqual(len(stats), 9)

    def test_different_batch_sizes(self):
        """Tests that the probe works with different batch sizes."""
        probe = ProbeNetwork(self.config)

        for batch_size in [1, 4, 8]:
            h_jepa_features = torch.randn(batch_size, 768, 7, 7)
            gwt_slots = torch.randn(batch_size, 4, 768)
            lnn_state = torch.randn(batch_size, 256)

            output = probe(h_jepa_features, gwt_slots, lnn_state)

            self.assertEqual(output.shape[0], batch_size)
            self.assertEqual(output.shape[1], self.config['output_dim'])

    def test_gradient_flow(self):
        """Tests that gradients flow properly through the probe."""
        probe = ProbeNetwork(self.config)

        # Create dummy inputs with requires_grad=True
        h_jepa_features = torch.randn(self.batch_size, 768, 7, 7, requires_grad=True)
        gwt_slots = torch.randn(self.batch_size, 4, 768, requires_grad=True)
        lnn_state = torch.randn(self.batch_size, 256, requires_grad=True)

        output = probe(h_jepa_features, gwt_slots, lnn_state)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist where expected
        # All inputs are detached in the probe implementation ("Freezed Encoder")
        self.assertIsNone(h_jepa_features.grad)
        self.assertIsNone(gwt_slots.grad)
        self.assertIsNone(lnn_state.grad)


if __name__ == '__main__':
    unittest.main()
