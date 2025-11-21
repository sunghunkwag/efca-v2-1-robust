import unittest
import torch
from efca.bottleneck.s_gwt import SGWT


class TestSGWT(unittest.TestCase):
    """
    Unit tests for the s-GWT (Slot-based Global Workspace Theory) Bottleneck module.
    """

    def setUp(self):
        """Set up test configuration."""
        self.config = {
            'num_slots': 4,
            'dim': 128,
            'iters': 3,
            'hidden_dim': 256
        }
        self.batch_size = 2
        self.num_inputs = 49  # e.g., 7x7 spatial features

    def test_initialization(self):
        """Tests that s-GWT can be initialized without errors."""
        try:
            model = SGWT(self.config)
            self.assertIsInstance(model, SGWT)
            self.assertEqual(model.num_slots, 4)
            self.assertEqual(model.dim, 128)
        except Exception as e:
            self.fail(f"SGWT initialization failed with an exception: {e}")

    def test_forward_pass(self):
        """Tests the forward pass of s-GWT."""
        model = SGWT(self.config)

        # Create dummy input (B, N, D)
        inputs = torch.randn(self.batch_size, self.num_inputs, self.config['dim'])

        slots = model(inputs)

        self.assertEqual(slots.shape, (self.batch_size, self.config['num_slots'], self.config['dim']))
        self.assertIsInstance(slots, torch.Tensor)

    def test_different_num_inputs(self):
        """Tests s-GWT with different numbers of input features."""
        model = SGWT(self.config)

        for num_inputs in [16, 49, 64, 100]:
            inputs = torch.randn(self.batch_size, num_inputs, self.config['dim'])
            slots = model(inputs)

            self.assertEqual(slots.shape[0], self.batch_size)
            self.assertEqual(slots.shape[1], self.config['num_slots'])
            self.assertEqual(slots.shape[2], self.config['dim'])

    def test_slot_attention_iterations(self):
        """Tests that slot attention performs the specified number of iterations."""
        for num_iters in [1, 3, 5]:
            config = self.config.copy()
            config['iters'] = num_iters
            model = SGWT(config)

            inputs = torch.randn(self.batch_size, self.num_inputs, self.config['dim'])
            slots = model(inputs)

            self.assertEqual(slots.shape, (self.batch_size, self.config['num_slots'], self.config['dim']))

    def test_different_slot_counts(self):
        """Tests s-GWT with different numbers of slots."""
        for num_slots in [2, 4, 8]:
            config = self.config.copy()
            config['num_slots'] = num_slots
            model = SGWT(config)

            inputs = torch.randn(self.batch_size, self.num_inputs, self.config['dim'])
            slots = model(inputs)

            self.assertEqual(slots.shape[1], num_slots)

    def test_gradient_flow(self):
        """Tests that gradients flow properly through s-GWT."""
        model = SGWT(self.config)

        inputs = torch.randn(self.batch_size, self.num_inputs, self.config['dim'], requires_grad=True)

        slots = model(inputs)
        loss = slots.sum()
        loss.backward()

        self.assertIsNotNone(inputs.grad)

        # Check that model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

    def test_slot_initialization_randomness(self):
        """Tests that slots are initialized with randomness."""
        model = SGWT(self.config)

        inputs = torch.randn(self.batch_size, self.num_inputs, self.config['dim'])

        # Run forward pass twice
        slots1 = model(inputs)
        slots2 = model(inputs)

        # Slots should be different due to random initialization
        self.assertFalse(torch.allclose(slots1, slots2))


if __name__ == '__main__':
    unittest.main()
