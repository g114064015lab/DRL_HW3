# HW3 Understanding Report

## HW3-1: Naive DQN Implementation for Static Mode

In this part, we implemented a Deep Q-Network (DQN) agent to solve a simple Gridworld environment.

### 1. Environment (Static Mode)
The static mode gridworld is a 4x4 grid where objects are placed at fixed positions:
- **Player (P)**: Starts at `(0, 3)` (Top-Right)
- **Goal (+)**: Fixed at `(0, 0)` (Top-Left), Reward: `+10`
- **Pit (-)**: Fixed at `(0, 1)`, Reward: `-10`
- **Wall (W)**: Fixed at `(1, 1)`, blocks movement.
- **Other cells**: Normal path, Reward: `-1` per step.

### 2. Basic DQN Architecture
The state is obtained using `render_np()`, which generates a 3D numpy array indicating the positions of the pieces. We flattened this array into a 1D tensor of size `64` to feed into a fully connected neural network (MLP).
- **Architecture**: `64 (Input)` -> `64 (ReLU)` -> `64 (ReLU)` -> `4 (Output Actions)`
- **Experience Replay Buffer**: Stabilizes training by breaking correlation in consecutive samples. We sample batches of 32 from a 1000-capacity buffer.

---

## HW3-2: Enhanced DQN Variants for Player Mode

In `player` mode, the player's starting position is randomized, making the environment more stochastic and harder to learn.

### 1. Double DQN
**Improvement**: Standard DQN suffers from "overestimation bias" because it uses `max(Q(s', a'))` from the same network used to select the action. Double DQN mitigates this by decoupling action selection and evaluation.
- **Action Selection**: Uses the Main Network to find the best action `argmax(Q_main(s', a'))`.
- **Action Evaluation**: Uses the Target Network to evaluate that specific action's value `Q_target(s', best_action)`.
This results in more stable and realistic Q-value estimates.

### 2. Dueling DQN
**Improvement**: In many states, the exact action taken doesn't heavily influence the outcome (e.g., when far away from the goal or pit). Dueling DQN splits the final layers into two streams:
- **Value Stream $V(s)$**: Estimates how good it is to be in state $s$.
- **Advantage Stream $A(s, a)$**: Estimates the advantage of taking action $a$ over other actions in state $s$.
They are aggregated via $Q(s, a) = V(s) + (A(s, a) - \text{mean}(A(s, a)))$. This allows the network to learn the value of states without needing to learn the effect of every action for every state, significantly speeding up training in stochastic starting states.

---

## HW3-3: Enhance DQN for Random Mode WITH Training Tips

In `random` mode, the Player, Goal, Pit, and Wall are all randomized, creating a highly complex state space. To solve this, we used **PyTorch Lightning** and added specific training tips.

### 1. PyTorch Lightning Integration
Converting the model to `pl.LightningModule` provides a robust, scalable framework. It naturally handles optimizer configurations, training steps, and metric logging without boilerplate code.

### 2. Training Tips
- **Gradient Clipping**: Random modes cause high variance in rewards (e.g., spawning immediately next to a pit). This can cause huge loss spikes and "exploding gradients". We apply `clip_gradients(norm=1.0)` to ensure weight updates stay bounded and stable.
- **Learning Rate Scheduling (`StepLR`)**: At the beginning of training, a higher learning rate allows rapid exploration. As epochs progress, `StepLR` decays the learning rate (e.g., $\gamma = 0.9$ every 200 epochs). This helps the network converge securely in the noisy, randomized state space rather than oscillating wildly.

---

## HW3-4: Rainbow DQN (Bonus)

To fully tackle the `random` mode Gridworld, we implemented a **Simplified Rainbow DQN**, bringing together multiple state-of-the-art enhancements:

### 1. Noisy Nets for Exploration
**Improvement**: Instead of using heuristic $\epsilon$-greedy exploration (randomly choosing actions), we replaced standard linear layers with `NoisyLinear` layers. These layers add parametric Gaussian noise to the weights and biases.
- The network "learns" how much noise (exploration) to inject based on the training loss. Over time, it naturally phases out noise as it becomes more confident in the Q-values.

### 2. Prioritized Experience Replay (PER)
**Improvement**: Standard Replay Buffers sample transitions uniformly. PER samples transitions based on their **TD-error** (Temporal Difference error).
- Transitions that "surprised" the network (high TD-error) have a higher probability of being sampled.
- We use Importance Sampling (IS) weights to correct for the sampling bias introduced by this prioritized replay.

### 3. Combining with Double & Dueling DQN
The Rainbow DQN effectively combines **Noisy Nets** and **PER** with the aforementioned **Double Q-learning** and **Dueling Network Architecture**. The combination of these techniques results in much faster convergence and more robust policies compared to any single enhancement alone.
