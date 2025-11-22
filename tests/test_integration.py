import unittest
import torch
import yaml
from unittest.mock import patch, MagicMock
from efca.perception.h_jepa import HJEPA
from efca.dynamics.ct_lnn import CTLNN
from efca.policy.task_policy import TaskPolicy
from efca.agent import EFCAgent

class TestIntegration(unittest.TestCase):
    """
    Integration tests for the complete Phase 0 agent.
    """

    def test_agent_assembly_and_step(self):
        """
        Tests that the Phase 0 agent can be assembled and run for a few steps.
        """
        try:
            # Load configuration
            with open('configs/default_config.yaml', 'r') as f:
                config = yaml.safe_load(f)

            # Initialize the agent's modules
            perception = HJEPA(config=config['h_jepa'])
            dynamics = CTLNN(config=config['ct_lnn'])
            policy = TaskPolicy(
                hidden_dim=config['task_policy']['hidden_dim'],
                action_dim=config['task_policy']['action_dim']
            )

            # Create a dummy input tensor
            dummy_input = torch.randn(1, 3, 224, 224)

            # --- Perform a few forward passes ---
            perception_loss, online_features = perception(dummy_input)
            perception_output = online_features.mean(dim=[2, 3])

            h = dynamics.init_state(batch_size=1)
            h = dynamics.forward(h, perception_output)
            dist, value = policy.forward(h)

            self.assertIsInstance(dist, torch.distributions.Categorical)
            self.assertEqual(value.shape, (1, 1))

        except Exception as e:
            self.fail(f"Agent integration test failed with an exception: {e}")
    
    def test_full_agent_phase0(self):
        """
        Tests the full EFCAgent in Phase 0 mode.
        """
        try:
            # Load configuration
            with open('configs/default_config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            config['task_policy']['action_dim'] = 4
            agent = EFCAgent(config)
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, 224, 224)
            
            # Forward pass
            result = agent(dummy_input)
            dist, value, h_new, meta_delta, perception_loss, probe_output = result
            
            # Check outputs
            self.assertIsInstance(dist, torch.distributions.Categorical)
            self.assertEqual(value.shape, (1, 1))
            self.assertIsNotNone(h_new)
            self.assertIsNone(meta_delta)  # Phase 0 has meta disabled
            self.assertIsNone(probe_output)  # Phase 0 has probe disabled
            
        except Exception as e:
            self.fail(f"Full agent Phase 0 test failed: {e}")
    
    def test_full_agent_phase1(self):
        """
        Tests the full EFCAgent in Phase 1 mode with meta-controller enabled.
        """
        try:
            # Load Phase 1 configuration
            with open('configs/phase1_config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            config['task_policy']['action_dim'] = 4
            agent = EFCAgent(config)
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, 224, 224)
            
            # Forward pass
            result = agent(dummy_input)
            dist, value, h_new, meta_delta, perception_loss, probe_output = result
            
            # Check outputs
            self.assertIsInstance(dist, torch.distributions.Categorical)
            self.assertEqual(value.shape, (1, 1))
            self.assertIsNotNone(h_new)
            self.assertIsNotNone(probe_output)  # Phase 1 has probe enabled
            self.assertEqual(probe_output.shape, (1, config['probe']['output_dim']))
            self.assertIsNotNone(meta_delta)  # Phase 1 has meta enabled
            
        except Exception as e:
            self.fail(f"Full agent Phase 1 test failed: {e}")
    
    @patch('efca.browser_interface.sync_playwright')
    def test_agent_with_browser(self, mock_playwright):
        """
        Tests the agent with browser integration enabled.
        """
        try:
            # Setup mocks
            mock_pw = MagicMock()
            mock_playwright.return_value.start.return_value = mock_pw
            mock_browser = MagicMock()
            mock_pw.chromium.launch.return_value = mock_browser
            mock_context = MagicMock()
            mock_browser.new_context.return_value = mock_context
            mock_page = MagicMock()
            mock_context.new_page.return_value = mock_page
            
            # Load configuration with browser enabled
            with open('configs/default_config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            config['enable_browser'] = True
            config['task_policy']['action_dim'] = 4
            agent = EFCAgent(config)
            
            # Check browser is initialized
            self.assertIsNotNone(agent.browser)
            
            # Test browser action
            result = agent.execute_browser_action('navigate', url='https://example.com')
            self.assertIsInstance(result, str)
            
            # Cleanup
            agent.cleanup()
            
        except Exception as e:
            self.fail(f"Agent with browser test failed: {e}")
    
    def test_device_transfer(self):
        """
        Tests that the agent can be moved between devices.
        """
        try:
            # Load configuration
            with open('configs/default_config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            config['task_policy']['action_dim'] = 4
            config['training']['device'] = 'cpu'
            agent = EFCAgent(config)
            
            # Create input on CPU
            dummy_input = torch.randn(1, 3, 224, 224)
            
            # Forward pass
            result = agent(dummy_input)
            dist, value, h_new, meta_delta, perception_loss, probe_output = result
            
            # Verify outputs are on CPU
            self.assertEqual(value.device.type, 'cpu')
            
            # Note: GPU test would require CUDA availability
            # if torch.cuda.is_available():
            #     agent = agent.to('cuda')
            #     dummy_input = dummy_input.to('cuda')
            #     result = agent(dummy_input)
            #     ...
            
        except Exception as e:
            self.fail(f"Device transfer test failed: {e}")

    def test_full_agent_phase0(self):
        """
        Tests the full EFCAgent in Phase 0 mode.
        """
        try:
            # Load configuration
            with open('configs/default_config.yaml', 'r') as f:
                config = yaml.safe_load(f)

            config['task_policy']['action_dim'] = 4
            agent = EFCAgent(config)

            # Create dummy input
            dummy_input = torch.randn(1, 3, 224, 224)

            # Forward pass
            result = agent(dummy_input)
            dist, value, h_new, meta_delta, perception_loss, probe_output = result

            # Check outputs
            self.assertIsInstance(dist, torch.distributions.Categorical)
            self.assertEqual(value.shape, (1, 1))
            self.assertIsNotNone(h_new)
            self.assertIsNone(meta_delta)  # Phase 0 has meta disabled
            self.assertIsNone(probe_output)  # Phase 0 has probe disabled

        except Exception as e:
            self.fail(f"Full agent Phase 0 test failed: {e}")

    def test_full_agent_phase1(self):
        """
        Tests the full EFCAgent in Phase 1 mode with meta-controller enabled.
        """
        try:
            # Load Phase 1 configuration
            with open('configs/phase1_config.yaml', 'r') as f:
                config = yaml.safe_load(f)

            config['task_policy']['action_dim'] = 4
            agent = EFCAgent(config)

            # Create dummy input
            dummy_input = torch.randn(1, 3, 224, 224)

            # Forward pass
            result = agent(dummy_input)
            dist, value, h_new, meta_delta, perception_loss, probe_output = result

            # Check outputs
            self.assertIsInstance(dist, torch.distributions.Categorical)
            self.assertEqual(value.shape, (1, 1))
            self.assertIsNotNone(h_new)
            self.assertIsNotNone(probe_output)  # Phase 1 has probe enabled
            self.assertEqual(probe_output.shape, (1, config['probe']['output_dim']))
            self.assertIsNotNone(meta_delta)  # Phase 1 has meta enabled

        except Exception as e:
            self.fail(f"Full agent Phase 1 test failed: {e}")

    @patch('efca.browser_interface.sync_playwright')
    def test_agent_with_browser(self, mock_playwright):
        """
        Tests the agent with browser integration enabled.
        """
        try:
            # Setup mocks
            mock_pw = MagicMock()
            mock_playwright.return_value.start.return_value = mock_pw
            mock_browser = MagicMock()
            mock_pw.chromium.launch.return_value = mock_browser
            mock_context = MagicMock()
            mock_browser.new_context.return_value = mock_context
            mock_page = MagicMock()
            mock_context.new_page.return_value = mock_page

            # Load configuration with browser enabled
            with open('configs/default_config.yaml', 'r') as f:
                config = yaml.safe_load(f)

            config['enable_browser'] = True
            config['task_policy']['action_dim'] = 4
            agent = EFCAgent(config)

            # Check browser is initialized
            self.assertIsNotNone(agent.browser)

            # Test browser action
            result = agent.execute_browser_action('navigate', url='https://example.com')
            self.assertIsInstance(result, str)

            # Cleanup
            agent.cleanup()

        except Exception as e:
            self.fail(f"Agent with browser test failed: {e}")

    def test_device_transfer(self):
        """
        Tests that the agent can be moved between devices.
        """
        try:
            # Load configuration
            with open('configs/default_config.yaml', 'r') as f:
                config = yaml.safe_load(f)

            config['task_policy']['action_dim'] = 4
            config['training']['device'] = 'cpu'
            agent = EFCAgent(config)

            # Create input on CPU
            dummy_input = torch.randn(1, 3, 224, 224)

            # Forward pass
            result = agent(dummy_input)
            dist, value, h_new, meta_delta, perception_loss, probe_output = result

            # Verify outputs are on CPU
            self.assertEqual(value.device.type, 'cpu')

            # Note: GPU test would require CUDA availability
            # if torch.cuda.is_available():
            #     agent = agent.to('cuda')
            #     dummy_input = dummy_input.to('cuda')
            #     result = agent(dummy_input)
            #     ...

        except Exception as e:
            self.fail(f"Device transfer test failed: {e}")

if __name__ == '__main__':
    unittest.main()
