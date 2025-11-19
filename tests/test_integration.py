import unittest
import torch
import yaml
from efca.perception.h_jepa import HJEPA
from efca.dynamics.ct_lnn import CTLNN
from efca.policy.task_policy import TaskPolicy
from efca.probe.probe_network import ProbeNetwork
from efca.metacontrol.meta_controller import MetaController

class TestIntegration(unittest.TestCase):
    """
    Integration tests for the complete Phase 1 agent.
    """

    def test_agent_assembly_and_step(self):
        """
        Tests that the Phase 1 agent can be assembled and run for a few steps.
        """
        try:
            # Load configuration
            with open('configs/default_config.yaml', 'r') as f:
                config = yaml.safe_load(f)

            # Initialize the agent's modules
            perception = HJEPA(embed_dim=config['h_jepa']['embed_dim'])
            dynamics = CTLNN(
                input_dim=config['ct_lnn']['input_dim'],
                hidden_dim=config['ct_lnn']['hidden_dim'],
                output_dim=config['ct_lnn']['output_dim']
            )
            policy = TaskPolicy(
                hidden_dim=config['task_policy']['hidden_dim'],
                action_dim=config['task_policy']['action_dim']
            )
            probe = ProbeNetwork(
                hidden_dim=config['ct_lnn']['hidden_dim'],
                probe_dim=config['probe_network']['probe_dim']
            )
            meta_controller = MetaController(
                probe_dim=config['probe_network']['probe_dim']
            )

            # Create a dummy input tensor
            dummy_input = torch.randn(1, 3, 224, 224)

            # --- Perform a few forward passes ---
            perception_loss, online_features = perception(dummy_input)
            perception_output = online_features.mean(dim=[2, 3])

            h = dynamics.init_state(batch_size=1)
            h = dynamics.forward(h, perception_output)

            probe_output = probe(h)
            meta_delta = meta_controller(probe_output)

            dist, value = policy.forward(h)

            self.assertIsInstance(dist, torch.distributions.Categorical)
            self.assertEqual(value.shape, (1, 1))
            self.assertLessEqual(meta_delta.abs().item(), config['meta_controller']['max_delta'] + 1e-9)

        except Exception as e:
            self.fail(f"Agent integration test failed with an exception: {e}")

if __name__ == '__main__':
    unittest.main()
