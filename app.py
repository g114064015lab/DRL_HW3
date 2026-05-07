import streamlit as st
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from Gridworld import Gridworld

# ----------------- Q-Network -----------------
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# ----------------- Replay Buffer -----------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.stack(state), action, reward, np.stack(next_state), done

    def __len__(self):
        return len(self.buffer)

# ----------------- Agent -----------------
class DQNAgent:
    def __init__(self, state_size, action_size, lr):
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayBuffer(1000)
        self.epsilon = 1.0
        self.gamma = 0.9

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def train_step(self, batch_size):
        if len(self.memory) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            max_next_q_values = self.q_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

# ----------------- Streamlit UI -----------------
st.title("DRL HW3: Naive DQN for Static Gridworld")
st.markdown("This application demonstrates a Naive DQN with Experience Replay Buffer solving the Static Gridworld environment.")

st.sidebar.header("Hyperparameters")
learning_rate = st.sidebar.number_input("Learning Rate", value=1e-3, format="%.4f")
epochs = st.sidebar.slider("Epochs", 100, 3000, 1000)
epsilon_decay = st.sidebar.slider("Epsilon Decay", 0.9, 0.999, 0.99, step=0.001)
batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)
max_steps = 50

if st.button("Start Training"):
    env = Gridworld(size=4, mode='static')
    action_map = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    
    state_size = env.board.render_np().size # 4x4x4 = 64
    action_size = 4
    
    agent = DQNAgent(state_size, action_size, learning_rate)
    agent.memory = ReplayBuffer(1000) # Ensure buffer resets
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    chart_empty = st.empty()
    
    rewards_history = []
    loss_history = []
    
    for epoch in range(epochs):
        env = Gridworld(size=4, mode='static')
        state = env.board.render_np().flatten()
        total_reward = 0
        loss_sum = 0
        loss_count = 0
        
        for step in range(max_steps):
            action_idx = agent.act(state)
            action = action_map[action_idx]
            
            env.makeMove(action)
            reward = env.reward()
            next_state = env.board.render_np().flatten()
            
            done = reward == 10 or reward == -10
            
            agent.memory.push(state, action_idx, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            loss = agent.train_step(batch_size)
            if loss is not None:
                loss_sum += loss
                loss_count += 1
                
            if done:
                break
                
        agent.epsilon = max(0.01, agent.epsilon * epsilon_decay)
        rewards_history.append(total_reward)
        if loss_count > 0:
            loss_history.append(loss_sum / loss_count)
        else:
            loss_history.append(0)
            
        if epoch % 20 == 0 or epoch == epochs - 1:
            progress_bar.progress((epoch + 1) / epochs)
            status_text.text(f"Epoch {epoch}/{epochs} | Reward: {total_reward} | Epsilon: {agent.epsilon:.2f}")
            
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].plot(rewards_history)
            ax[0].set_title("Total Reward")
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("Reward")
            
            ax[1].plot(loss_history, color='orange')
            ax[1].set_title("Average Loss")
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("Loss")
            
            chart_empty.pyplot(fig)
            plt.close(fig)
            
    progress_bar.progress(1.0)
    status_text.text("Training Finished!")
    
    st.subheader("Test Run (Greedy Policy)")
    env = Gridworld(size=4, mode='static')
    state = env.board.render_np().flatten()
    agent.epsilon = 0.0 # pure greedy
    
    test_steps = []
    st.text("Initial state:")
    st.text(str(env.display()))
    for step in range(max_steps):
        action_idx = agent.act(state)
        action = action_map[action_idx]
        env.makeMove(action)
        reward = env.reward()
        done = reward == 10 or reward == -10
        test_steps.append((env.display(), action, reward))
        if done:
            break
        state = env.board.render_np().flatten()
        
    for i, (board_display, action, reward) in enumerate(test_steps):
        st.write(f"**Step {i+1}**: Action `{action}`, Reward: `{reward}`")
        st.text(str(board_display))
        if reward == 10:
            st.success("Goal Reached!")
        elif reward == -10:
            st.error("Fell into Pit!")
