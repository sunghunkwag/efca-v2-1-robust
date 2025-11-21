import gymnasium as gym
import numpy as np
import sys

def verify_mujoco():
    print("Verifying MuJoCo integration...")
    try:
        # Try to make the environment
        env = gym.make("Ant-v4", render_mode="rgb_array")
        print("Successfully created Ant-v4 environment.")

        # Reset
        obs, info = env.reset()
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")

        # Step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("Successfully ran one step.")

        env.close()
        print("MuJoCo verification PASSED.")
        return True
    except ImportError as e:
        print(f"ImportError: {e}")
        print("Please ensure gymnasium[mujoco] is installed.")
        return False
    except Exception as e:
        print(f"MuJoCo verification FAILED: {e}")
        return False

if __name__ == "__main__":
    success = verify_mujoco()
    sys.exit(0 if success else 1)
