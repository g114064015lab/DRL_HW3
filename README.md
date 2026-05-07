# DRL HW3: Naive DQN for Static Gridworld

**Streamlit Demo URL**: [https://drlhw3-7junkaslnuc45wby5vx9ye.streamlit.app/](https://drlhw3-7junkaslnuc45wby5vx9ye.streamlit.app/)

## Overview
This repository contains the implementation of a Deep Q-Network (DQN) and its variants to solve a Gridworld environment. The application is deployed and visualized using Streamlit.

## Acknowledgements / Citations
The Gridworld environment files (`Gridworld.py` and `GridBoard.py`) are referenced and used from:
- [DeepReinforcementLearningInAction/tree/master](https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/tree/master)

## Assignment Details

### HW3-1: Naive DQN for Static Mode
- **Environment**: Static mode (fixed positions for all pieces).
- **Implementation**: Basic DQN with PyTorch.
- **Components**: Experience Replay Buffer and Epsilon-Greedy Exploration.

### HW3-2: Enhanced DQN Variants for Player Mode
- **Environment**: Player mode (Player spawns randomly, other pieces fixed).
- **Double DQN**: Introduces a Target Network to evaluate the best action proposed by the Main Network, reducing the overestimation bias of Q-values.
- **Dueling DQN**: Splits the network architecture into a Value stream (estimating the state value) and an Advantage stream (estimating the relative advantage of actions), which helps the agent learn which states are valuable independent of actions.

### HW3-3: Enhance DQN for Random Mode WITH Training Tips
- **Environment**: Random mode (all pieces spawn in random locations).
- **Framework Conversion**: Converted the DQN training pipeline into a **PyTorch Lightning** module (`pl.LightningModule`), offering a more structured training loop and easy integrations.
- **Training Tips**: 
  - **Learning Rate Scheduling**: Applied `StepLR` to decay the learning rate as epochs progress, helping to converge fine-grained details later in training.
  - **Gradient Clipping**: Applied gradient clipping (`gradient_clip_val=1.0`) to prevent exploding gradients and stabilize the neural network updates when facing random, high-variance states.

## Running Locally
To run this application locally, ensure you have the necessary dependencies installed:

```bash
pip install -r requirements.txt
streamlit run app.py
```
