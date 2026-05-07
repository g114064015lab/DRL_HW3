# DRL HW3: Naive DQN for Static Gridworld

**Streamlit Demo URL**: [https://drlhw3-7junkaslnuc45wby5vx9ye.streamlit.app/](https://drlhw3-7junkaslnuc45wby5vx9ye.streamlit.app/)

## Overview
This repository contains the implementation of a Deep Q-Network (DQN) with an Experience Replay Buffer, designed to solve a simple Static Gridworld environment. The application is deployed and visualized using Streamlit.

## Acknowledgements / Citations
The Gridworld environment files (`Gridworld.py` and `GridBoard.py`) are referenced and used from:
- [DeepReinforcementLearningInAction/tree/master](https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/tree/master)

## Assignment Details (HW3-1)
- Implemented a basic Naive DQN algorithm using PyTorch.
- Added an Experience Replay Buffer to stabilize training.
- Solved the environment under `static` mode (fixed positions for Player, Goal, Pit, and Wall).
- Deployed a web interface for training and inference using Streamlit.

## Running Locally
To run this application locally, ensure you have the necessary dependencies installed:

```bash
pip install -r requirements.txt
streamlit run app.py
```
