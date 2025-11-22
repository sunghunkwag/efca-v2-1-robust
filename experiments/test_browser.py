import sys
import os

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from efca.agent import EFCAgent

def test_browser_integration():
    config = {
        'phase': 0,
        'h_jepa': {'embed_dim': 64},
        'bottleneck': {'num_slots': 4, 'slot_dim': 64},
        'ct_lnn': {'input_dim': 64, 'hidden_dim': 64, 'output_dim': 64},
        'task_policy': {'hidden_dim': 64, 'action_dim': 4},
        'training': {'device': 'cpu'},
        'enable_browser': True
    }

    print("Initializing EFCAgent with browser enabled...")
    agent = EFCAgent(config)

    print("Testing navigation...")
    result = agent.execute_browser_action('navigate', url='https://www.example.com')
    print(result)

    print("Testing get_title...")
    title = agent.execute_browser_action('get_title')
    print(f"Title: {title}")

    print("Testing screenshot...")
    screenshot_path = os.path.join(os.path.dirname(__file__), 'test_screenshot.png')
    result = agent.execute_browser_action('screenshot', path=screenshot_path)
    print(result)

    print("Closing browser...")
    agent.browser.close()
    print("Browser closed.")

if __name__ == "__main__":
    test_browser_integration()
