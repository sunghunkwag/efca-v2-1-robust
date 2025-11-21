# EFCA-v2.1: Robust Functional Consciousness Agent

This repository contains the implementation of the EFCA-v2.1, a feasibility-focused redesign of a functional consciousness agent. The primary objective is to implement measurable functional self-awareness with a strong emphasis on training stability and incremental complexity.

## Specification

The detailed specification for this project can be found in the [efca-v2-1-robust-english.md](efca-v2-1-robust-english.md) file.

## Architecture Overview

EFCA-v2.1 is designed with a "Stability-First" approach, incorporating a Stabilization Layer to manage the interaction between its Fast and Slow loops. The core modules of the architecture are:

- **H-JEPA (Perception):** A Hierarchical Joint-Embedding Predictive Architecture that uses predictive coding. It employs a strict Stop-Gradient on targets to prevent representation collapse.
- **s-GWT (Bottleneck):** A Global Workspace that ensures information competition is tractable by starting with a reduced number of initial slots.
- **Hybrid CT-LNN (Dynamics):** A Continuous-Time Liquid Neural Network that supports both a Discrete Mode for stable initial training and an ODE Mode for advanced temporal modeling.
- **Task Policy (Action):** A standard Actor-Critic model for action generation.
- **Clamped Meta-Controller (Self-Regulation):** A controller that outputs relative updates with hard clipping to prevent parameter explosion and ensure stability.
- **Probe Network (Metacognition):** A network that monitors the internal states of the system.

## Phased Implementation Roadmap

The implementation of EFCA-v2.1 is divided into four distinct phases to ensure a stable and incremental development process:

### Phase 0: The "Zombie" Agent (Baseline)
- **Configuration:** H-JEPA + CT-LNN (Discrete) + Task Policy.
- **Meta-Controller:** Disabled (Fixed parameters).
- **Goal:** Verify that the core components can solve a basic task using standard reinforcement learning.

### Phase 1: The "Curious" Agent (Single-Axis Meta)
- **Configuration:** Enable Probe + Meta-Controller.
- **Action Space:** The Meta-Controller will only control `epsilon_explore`.
- **Goal:** Verify that `epsilon` increases when the JEPA Error (uncertainty) is high.

### Phase 2: The "Self-Tuning" Agent (Full Meta-Control)
- **Configuration:** Enable control of layer weights and the learning rate.
- **Constraint:** Use "Clamped Delta Control" with a maximum 5% change per step.
- **Goal:** Demonstrate faster convergence on non-stationary tasks.

### Phase 3: The "Fluid" Agent (Full ODE + GWT)
- **Configuration:** Switch the CT-LNN to the full ODE solver and enable full GWT sparsity.
- **Goal:** Achieve advanced temporal modeling and memory integration.

## Installation

To install the necessary dependencies, first clone the repository and then run the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Training

The training scripts are located in the `experiments/` directory.

**Phase 0 (Zombie Agent):**
```bash
python experiments/train.py --config configs/default_config.yaml
```

**Phase 1 (Curious Agent with Meta-Controller):**
```bash
python experiments/train.py --config configs/phase1_config.yaml
```

### Browser Integration

To enable browser automation in your agent, set `enable_browser: true` in your configuration file:

```yaml
enable_browser: true
browser:
  headless: false  # Set to true for headless mode
  timeout: 30000   # Timeout in milliseconds
```

**Example browser usage:**
```python
from efca.agent import EFCAgent

config = {
    'enable_browser': True,
    'browser': {'headless': False, 'timeout': 30000},
    # ... other config settings
}

agent = EFCAgent(config)

# Navigate to URL
result = agent.execute_browser_action('navigate', url='https://example.com')

# Wait for element
agent.browser.wait_for_selector('button#submit', state='visible')

# Click element
result = agent.execute_browser_action('click', selector='button#submit')

# Type text
result = agent.execute_browser_action('type', selector='input#search', text='query')

# Take screenshot
result = agent.execute_browser_action('screenshot', path='screenshot.png')

# Cleanup
agent.cleanup()
```

## Testing

### Running All Tests

```bash
# Using unittest
python -m unittest discover -s tests -p "test_*.py" -v

# Or using pytest (if installed)
pytest tests/ -v
```

### Running Individual Test Modules

```bash
# Test H-JEPA perception module
python -m unittest tests.test_h_jepa

# Test CT-LNN dynamics module
python -m unittest tests.test_ct_lnn

# Test Task Policy module
python -m unittest tests.test_task_policy

# Test s-GWT bottleneck module
python -m unittest tests.test_s_gwt

# Test Probe Network
python -m unittest tests.test_probe_network

# Test Browser Interface
python -m unittest tests.test_browser_interface

# Test full integration
python -m unittest tests.test_integration
```

### Testing Browser Integration

```bash
python experiments/test_browser.py
```

## API Documentation

### Core Modules

#### H-JEPA (Perception)
- **Location:** `efca/perception/h_jepa.py`
- **Purpose:** Hierarchical Joint-Embedding Predictive Architecture for perception
- **Key Methods:**
  - `forward(x)`: Returns perception loss and online features
  - `update_target_encoder(tau)`: Updates target encoder using EMA

#### s-GWT (Bottleneck)
- **Location:** `efca/bottleneck/s_gwt.py`
- **Purpose:** Slot-based Global Workspace Theory for information routing
- **Key Methods:**
  - `forward(inputs)`: Returns attention slots (B, Num_Slots, Dim)

#### CT-LNN (Dynamics)
- **Location:** `efca/dynamics/ct_lnn.py`
- **Purpose:** Continuous-Time Liquid Neural Network for temporal dynamics
- **Key Methods:**
  - `init_state(batch_size)`: Initialize hidden state
  - `forward(h, x)`: Process input and return next state

#### Probe Network (Metacognition)
- **Location:** `efca/probe/probe_network.py`
- **Purpose:** Monitor internal states for meta-controller
- **Key Methods:**
  - `forward(h_jepa_features, gwt_slots, lnn_state)`: Returns probe output
  - `get_statistics(...)`: Extract statistical information

#### Task Policy (Action)
- **Location:** `efca/policy/task_policy.py`
- **Purpose:** Actor-Critic policy for action generation
- **Key Methods:**
  - `forward(h)`: Returns action distribution and value estimate

#### Browser Controller
- **Location:** `efca/browser_interface.py`
- **Purpose:** Browser automation interface
- **Key Methods:**
  - `navigate(url)`: Navigate to URL
  - `click(selector)`: Click element
  - `type(selector, text)`: Type text into element
  - `wait_for_selector(selector, state)`: Wait for element state
  - `press_key(key)`: Press keyboard key
  - `is_visible(selector)`: Check element visibility

## Troubleshooting

### Common Issues

**1. Browser automation fails**
```
Error: Browser not found
```
**Solution:** Install Playwright browsers:
```bash
python -m playwright install chromium
```

**2. CUDA out of memory**
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch size in configuration or use CPU:
```yaml
training:
  device: 'cpu'
```

**3. Module import errors**
```
ModuleNotFoundError: No module named 'efca'
```
**Solution:** Ensure you're in the project root directory and PYTHONPATH is set correctly.

**4. Checkpoint loading fails**
```
RuntimeError: Error loading checkpoint
```
**Solution:** Ensure checkpoint was saved with same model architecture. Delete checkpoints folder to start fresh.

### Performance Tips

1. **Use GPU acceleration**: Set `device: 'cuda'` in config if GPU is available
2. **Adjust batch size**: Increase for better GPU utilization, decrease if OOM errors occur
3. **Tune learning rate**: Start with 0.001 and adjust based on convergence
4. **Save checkpoints regularly**: Set `save_interval` to save progress

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.
