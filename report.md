# HW3-1 Understanding Report

## Naive DQN Implementation for Static Mode

In this assignment, we implemented a Deep Q-Network (DQN) agent to solve a simple Gridworld environment.

### 1. Environment (Static Mode)
The static mode gridworld is a 4x4 grid where objects are placed at fixed positions:
- **Player (P)**: Starts at `(0, 3)` (Top-Right)
- **Goal (+)**: Fixed at `(0, 0)` (Top-Left), Reward: `+10`
- **Pit (-)**: Fixed at `(0, 1)`, Reward: `-10`
- **Wall (W)**: Fixed at `(1, 1)`, blocks movement.
- **Other cells**: Normal path, Reward: `-1` per step.

### 2. Basic DQN Architecture
The state is obtained using `render_np()`, which generates a 3D numpy array indicating the positions of the pieces. We flattened this array into a 1D tensor of size `64` (4 channels * 4x4 grid) to feed into a fully connected neural network (MLP).
The Q-network architecture:
- **Input layer**: `64` nodes (flattened state)
- **Hidden layer 1**: `64` nodes with ReLU activation
- **Hidden layer 2**: `64` nodes with ReLU activation
- **Output layer**: `4` nodes corresponding to the Q-values of 4 possible actions: `Up`, `Down`, `Left`, `Right`.

### 3. Experience Replay Buffer
To stabilize the DQN training and break the correlation between consecutive samples, we used an **Experience Replay Buffer**.
- **Mechanism**: The agent stores its experiences `(state, action, reward, next_state, done)` in a finite-sized queue (capacity: 1000).
- **Sampling**: During the training step, a random mini-batch of 32 experiences is sampled from the buffer. This allows the network to learn from past experiences multiple times and reduces variance in the gradient updates.

### 4. Training Process
- We used an **$\epsilon$-greedy policy** to balance exploration and exploitation. Epsilon starts at `1.0` (pure exploration) and decays gradually to `0.01` (mostly exploitation).
- The network updates its weights using **MSE Loss** between the predicted Q-values and the Target Q-values: `Target = Reward + Gamma * max(Q(next_state))`.
- After roughly 1000 epochs, the agent successfully learns the optimal policy to avoid the pit, navigate around the wall, and reach the goal efficiently.
