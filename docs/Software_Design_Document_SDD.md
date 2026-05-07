# Software Design Document (SDD)

**Project Name**: Deep Reinforcement Learning HW3 - DQN Variants for Gridworld
**Version**: 1.0.0
**Date**: May 2026

---

## 1. Introduction

### 1.1 Purpose
This Software Design Document (SDD) provides a comprehensive architectural overview of the Deep Reinforcement Learning (DRL) application built for solving various modes of a Gridworld environment. It outlines the system structure, data flow, component designs, and UI integrations.

### 1.2 Scope
The application encompasses an interactive Streamlit frontend that triggers deep reinforcement learning training loops using PyTorch and PyTorch Lightning. It demonstrates the progression from a Naive Deep Q-Network (DQN) to a state-of-the-art Rainbow DQN.

---

## 2. System Architecture

The system follows a Model-View-Controller (MVC) inspired architecture tailored for data science workflows:
- **View (Frontend)**: `streamlit` application rendering hyperparameters and dynamic HTML tables.
- **Controller (Logic)**: Streamlit callbacks acting as orchestration scripts managing the RL loop and UI states.
- **Model (Backend)**: `Gridworld` environment simulators and PyTorch-based neural networks (`QNetwork`, `DuelingQNetwork`, `NoisyLinear`).

### Directory Structure
```
DRL_HW3/
├── app.py                # Main Streamlit application and agent orchestrator
├── Gridworld.py          # Environment dynamics
├── GridBoard.py          # Environment rendering logic
├── requirements.txt      # Python dependencies
├── start.sh              # Initialization and daemon execution script
├── ending.sh             # Graceful shutdown and archiving script
├── docs/                 # Documentation (SDD, Reports, Conversation Logs)
└── logs/ models/ results/ # Dynamically generated folders
```

---

## 3. Component Design

### 3.1 Environment (`Gridworld`)
A customized 4x4 grid. The state is represented as a 3D numpy array flattened into a 64-dimensional feature vector.
- **Static Mode**: All pieces (Player, Goal, Pit, Wall) are fixed.
- **Player Mode**: The Player spawns randomly, other pieces are fixed.
- **Random Mode**: All pieces spawn randomly.

### 3.2 DQN Variants
The project incrementally builds upon the base DQN:

1. **Naive DQN** (`DQNAgent`):
   - Uses a single standard Multi-Layer Perceptron (MLP) for both action selection and value evaluation.
   - High risk of overestimation bias.

2. **Double & Dueling DQN** (`EnhancedDQNAgent`):
   - **Double DQN**: Introduces a `target_network`. Action is selected by the `q_network`, but its Q-value is evaluated by the `target_network`.
   - **Dueling DQN**: Splits the architecture into a Value stream and an Advantage stream, aggregating them to compute final Q-values. Highly effective in noisy environments where the state value dominates the action advantage.

3. **PyTorch Lightning Integration** (`LightningDQN`):
   - Wraps the neural network, loss function, and optimizers into a `pl.LightningModule`.
   - Integrates advanced training techniques such as Gradient Clipping (`clip_grad_norm_`) and Learning Rate Scheduling (`StepLR`).

4. **Rainbow DQN** (`RainbowDQNAgent`):
   - **Noisy Nets**: Replaces $\epsilon$-greedy exploration with `NoisyLinear` layers, injecting Gaussian noise directly into network parameters for self-driven exploration.
   - **Prioritized Experience Replay (PER)**: Replaces uniform sampling with TD-error-based prioritization, ensuring the agent learns faster from "surprising" transitions.

---

## 4. Data Flow

1. **Initialization**: User selects a mode (HW3-1 to HW3-4) and adjusts hyperparameters via Streamlit Sidebar.
2. **Environment Step**: The Agent observes `state_t`, uses the neural network to predict `q_values`, and selects `action_t`.
3. **Transition Storage**: The environment returns `reward_t` and `state_{t+1}`. The tuple `(s, a, r, s', done)` is pushed to the `ReplayBuffer` or `PrioritizedReplayBuffer`.
4. **Optimization**:
   - A batch is sampled from the buffer.
   - Bellman Target is calculated: $Y = R + \gamma \max Q_{\text{target}}(s')$.
   - Loss (MSE) is computed against predicted Q-values.
   - Backpropagation updates network weights.
5. **UI Update**: `matplotlib` charts (Reward/Loss curves) and `st.progress` are updated every $N$ epochs synchronously.

---

## 5. Deployment and Execution

### Local Shell Scripts
- **`start.sh`**: Handles pre-flight checks, directory creation, dependency verification, and daemonizes the Streamlit server using `nohup`.
- **`ending.sh`**: Safely terminates the background daemon via PID tracking, organizes generated logs/results, compresses them into an archive, and flushes python caches to guarantee a clean state.

### UI Rendering
The "Test Run" visualization avoids basic terminal text formatting. Instead, it utilizes `st.markdown(unsafe_allow_html=True)` to dynamically generate custom CSS-styled HTML grids. Combining this with `time.sleep`, it provides an authentic animation loop of the agent's greedy policy evaluation without requiring external frontend frameworks.
