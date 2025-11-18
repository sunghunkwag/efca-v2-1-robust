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

The training scripts are located in the `experiments/` directory. To run an experiment, you can use the following command:

```bash
python experiments/train.py --config configs/config.yaml
```
