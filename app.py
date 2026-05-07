import streamlit as st
import numpy as np
import random
import time
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from Gridworld import Gridworld

# ----------------- UI Styling & Helper -----------------
def render_board_html(board_array):
    html = '<table style="border-collapse: collapse; text-align: center; margin-bottom: 20px;">'
    for row in board_array:
        html += '<tr>'
        for cell in row:
            color = "white"
            if cell == 'P': color = "#4CAF50; color: white;" # Green Player
            elif cell == '+': color = "#FFD700; color: black;" # Gold Goal
            elif cell == '-': color = "#F44336; color: white;" # Red Pit
            elif cell == 'W': color = "#9E9E9E; color: white;" # Grey Wall
            else: color = "#f0f2f6; color: black;" # Empty cell
            html += f'<td style="width: 100px; height: 100px; border: 2px solid #ccc; background: {color} font-weight: bold; font-size: 40px;">{cell}</td>'
        html += '</tr>'
    html += '</table>'
    return html

# ----------------- Components for Rainbow -----------------
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
            
        probs = prios ** self.alpha
        if probs.sum() > 0:
            probs /= probs.sum()
        else:
            probs = np.ones(len(self.buffer)) / len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        
        state, action, reward, next_state, done = zip(*samples)
        return np.stack(state), action, reward, np.stack(next_state), done, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio
            
    def __len__(self):
        return len(self.buffer)

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

# ----------------- Networks -----------------
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

class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.value_stream = nn.Linear(64, 1)
        self.advantage_stream = nn.Linear(64, action_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

class RainbowQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(RainbowQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.value_stream = NoisyLinear(64, 1)
        self.advantage_stream = NoisyLinear(64, action_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals
        
    def reset_noise(self):
        self.value_stream.reset_noise()
        self.advantage_stream.reset_noise()

# ----------------- Agents -----------------
class DQNAgent:
    """HW3-1 Naive DQN"""
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
        if len(self.memory) < batch_size: return None
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

class EnhancedDQNAgent:
    """HW3-2 Double/Dueling DQN with Target Network"""
    def __init__(self, state_size, action_size, lr, is_dueling=False, is_double=False):
        self.state_size = state_size
        self.action_size = action_size
        self.is_double = is_double
        self.is_dueling = is_dueling
        
        if is_dueling:
            self.q_network = DuelingQNetwork(state_size, action_size)
            self.target_network = DuelingQNetwork(state_size, action_size)
        else:
            self.q_network = QNetwork(state_size, action_size)
            self.target_network = QNetwork(state_size, action_size)
            
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayBuffer(1000)
        self.epsilon = 1.0
        self.gamma = 0.9

    def act(self, state):
        if random.random() < self.epsilon: return random.randint(0, self.action_size - 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()

    def train_step(self, batch_size):
        if len(self.memory) < batch_size: return None
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        q_values = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            if self.is_double:
                best_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
                max_next_q_values = self.target_network(next_states).gather(1, best_actions)
            else:
                max_next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

class RainbowDQNAgent:
    """HW3-4 Simplified Rainbow DQN (Double + Dueling + PER + Noisy)"""
    def __init__(self, state_size, action_size, lr):
        self.state_size = state_size
        self.action_size = action_size
        
        self.q_network = RainbowQNetwork(state_size, action_size)
        self.target_network = RainbowQNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = PrioritizedReplayBuffer(1000)
        self.gamma = 0.9
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()

    def train_step(self, batch_size):
        if len(self.memory) < batch_size: return None
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(batch_size, self.beta)
        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        weights = torch.FloatTensor(weights).unsqueeze(1)

        self.q_network.reset_noise()
        self.target_network.reset_noise()

        q_values = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            best_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
            max_next_q_values = self.target_network(next_states).gather(1, best_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        td_errors = torch.abs(q_values - target_q_values).detach().numpy().flatten()
        self.memory.update_priorities(indices, td_errors + 1e-5)

        loss = (weights * (q_values - target_q_values)**2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# ----------------- PyTorch Lightning Module -----------------
class LightningDQN(pl.LightningModule):
    """HW3-3 DQN for random mode with training tips"""
    def __init__(self, state_size, action_size, lr, max_steps, epsilon_decay):
        super().__init__()
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.lr = lr
        self.max_steps = max_steps
        self.loss_fn = nn.MSELoss()
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        
        self.memory = ReplayBuffer(1000)
        self.env = Gridworld(size=4, mode='random')
        self.action_map = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
        self.last_epoch_reward = 0
        self.automatic_optimization = False 

    def configure_optimizers(self):
        optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
        return [optimizer], [scheduler]
        
    def act(self, state):
        if random.random() < self.epsilon: return random.randint(0, 3)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        
        state = self.env.board.render_np().flatten()
        total_reward = 0
        loss_sum = 0
        loss_count = 0
        
        for step in range(self.max_steps):
            action_idx = self.act(state)
            action = self.action_map[action_idx]
            self.env.makeMove(action)
            reward = self.env.reward()
            next_state = self.env.board.render_np().flatten()
            done = reward == 10 or reward == -10
            
            self.memory.push(state, action_idx, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if len(self.memory) >= 32:
                states, actions, rewards, next_states, dones = self.memory.sample(32)
                states = torch.FloatTensor(states).to(self.device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
                next_states = torch.FloatTensor(next_states).to(self.device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
                
                q_values = self.q_network(states).gather(1, actions)
                with torch.no_grad():
                    max_next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
                    target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
                    
                loss = self.loss_fn(q_values, target_q_values)
                opt.zero_grad()
                self.manual_backward(loss)
                self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
                opt.step()
                
                loss_sum += loss.item()
                loss_count += 1
                
            if done: break
                
        sch.step()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.last_epoch_reward = total_reward
        
        if self.current_epoch % 10 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        avg_loss = loss_sum / loss_count if loss_count > 0 else 0.0
        self.log('train_loss', avg_loss, prog_bar=True)
        self.env = Gridworld(size=4, mode='random')
        return None

    def train_dataloader(self):
        dataset = torch.utils.data.TensorDataset(torch.zeros(1))
        return torch.utils.data.DataLoader(dataset, batch_size=1)

class StreamlitLightningCallback(pl.Callback):
    def __init__(self, progress_bar, status_text, chart_empty, max_epochs):
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.chart_empty = chart_empty
        self.max_epochs = max_epochs
        self.rewards_history = []
        self.loss_history = []
        
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        loss = trainer.callback_metrics.get("train_loss", torch.tensor(0.0)).item()
        reward = pl_module.last_epoch_reward
        
        self.rewards_history.append(reward)
        self.loss_history.append(loss)
        
        if epoch % 20 == 0 or epoch == self.max_epochs - 1:
            self.progress_bar.progress((epoch + 1) / self.max_epochs)
            self.status_text.text(f"Epoch {epoch}/{self.max_epochs} | Reward: {reward} | Epsilon: {pl_module.epsilon:.2f} | LR: {trainer.optimizers[0].param_groups[0]['lr']:.5f}")
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].plot(self.rewards_history)
            ax[0].set_title("Total Reward")
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("Reward")
            ax[1].plot(self.loss_history, color='orange')
            ax[1].set_title("Average Loss")
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("Loss")
            self.chart_empty.pyplot(fig)
            plt.close(fig)

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="DRL HW3: DQN Variants", layout="wide")

assignment_part = st.sidebar.selectbox("Assignment Part", [
    "HW3-1: Naive DQN (Static Mode)",
    "HW3-2: Enhanced DQN (Player Mode)",
    "HW3-3: PyTorch Lightning (Random Mode)",
    "HW3-4: Rainbow DQN (Random Mode)"
])

st.title(f"DRL {assignment_part}")

st.sidebar.header("Hyperparameters")
learning_rate = st.sidebar.number_input("Learning Rate", value=1e-3, format="%.4f")
epochs = st.sidebar.slider("Epochs", 100, 3000, 1000)
epsilon_decay = st.sidebar.slider("Epsilon Decay", 0.9, 0.999, 0.99, step=0.001)
batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)
max_steps = 50

is_double = False
is_dueling = False
if "HW3-2" in assignment_part:
    st.sidebar.subheader("HW3-2 Options")
    is_double = st.sidebar.checkbox("Use Double DQN", value=True)
    is_dueling = st.sidebar.checkbox("Use Dueling DQN", value=True)

if st.button("Start Training"):
    action_map = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    action_size = 4
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    chart_empty = st.empty()
    
    if "HW3-1" in assignment_part:
        env = Gridworld(size=4, mode='static')
        state_size = env.board.render_np().size
        agent = DQNAgent(state_size, action_size, learning_rate)
        rewards_history = []
        loss_history = []
        for epoch in range(epochs):
            env = Gridworld(size=4, mode='static')
            state = env.board.render_np().flatten()
            total_reward, loss_sum, loss_count = 0, 0, 0
            for step in range(max_steps):
                action_idx = agent.act(state)
                env.makeMove(action_map[action_idx])
                reward = env.reward()
                next_state = env.board.render_np().flatten()
                done = reward == 10 or reward == -10
                agent.memory.push(state, action_idx, reward, next_state, done)
                state = next_state
                total_reward += reward
                loss = agent.train_step(batch_size)
                if loss is not None:
                    loss_sum += loss; loss_count += 1
                if done: break
            agent.epsilon = max(0.01, agent.epsilon * epsilon_decay)
            rewards_history.append(total_reward)
            loss_history.append(loss_sum / loss_count if loss_count > 0 else 0)
            if epoch % 20 == 0 or epoch == epochs - 1:
                progress_bar.progress((epoch + 1) / epochs)
                status_text.text(f"Epoch {epoch}/{epochs} | Reward: {total_reward} | Epsilon: {agent.epsilon:.2f}")
                fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                ax[0].plot(rewards_history); ax[0].set_title("Total Reward")
                ax[1].plot(loss_history, color='orange'); ax[1].set_title("Average Loss")
                chart_empty.pyplot(fig); plt.close(fig)
                
    elif "HW3-2" in assignment_part:
        env = Gridworld(size=4, mode='player')
        state_size = env.board.render_np().size
        agent = EnhancedDQNAgent(state_size, action_size, learning_rate, is_dueling=is_dueling, is_double=is_double)
        rewards_history = []
        loss_history = []
        for epoch in range(epochs):
            env = Gridworld(size=4, mode='player')
            state = env.board.render_np().flatten()
            total_reward, loss_sum, loss_count = 0, 0, 0
            for step in range(max_steps):
                action_idx = agent.act(state)
                env.makeMove(action_map[action_idx])
                reward = env.reward()
                next_state = env.board.render_np().flatten()
                done = reward == 10 or reward == -10
                agent.memory.push(state, action_idx, reward, next_state, done)
                state = next_state
                total_reward += reward
                loss = agent.train_step(batch_size)
                if loss is not None:
                    loss_sum += loss; loss_count += 1
                if done: break
            agent.epsilon = max(0.01, agent.epsilon * epsilon_decay)
            if epoch % 10 == 0: agent.update_target_network()
            rewards_history.append(total_reward)
            loss_history.append(loss_sum / loss_count if loss_count > 0 else 0)
            if epoch % 20 == 0 or epoch == epochs - 1:
                progress_bar.progress((epoch + 1) / epochs)
                status_text.text(f"Epoch {epoch}/{epochs} | Reward: {total_reward} | Epsilon: {agent.epsilon:.2f}")
                fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                ax[0].plot(rewards_history); ax[0].set_title("Total Reward")
                ax[1].plot(loss_history, color='orange'); ax[1].set_title("Average Loss")
                chart_empty.pyplot(fig); plt.close(fig)
                
    elif "HW3-3" in assignment_part:
        env = Gridworld(size=4, mode='random')
        state_size = env.board.render_np().size
        pl_agent = LightningDQN(state_size, action_size, learning_rate, max_steps, epsilon_decay)
        cb = StreamlitLightningCallback(progress_bar, status_text, chart_empty, epochs)
        trainer = pl.Trainer(max_epochs=epochs, callbacks=[cb], enable_progress_bar=False, enable_model_summary=False, logger=False)
        trainer.fit(pl_agent)
        agent = pl_agent
        
    elif "HW3-4" in assignment_part:
        env = Gridworld(size=4, mode='random')
        state_size = env.board.render_np().size
        agent = RainbowDQNAgent(state_size, action_size, learning_rate)
        rewards_history = []
        loss_history = []
        for epoch in range(epochs):
            env = Gridworld(size=4, mode='random')
            state = env.board.render_np().flatten()
            total_reward, loss_sum, loss_count = 0, 0, 0
            for step in range(max_steps):
                action_idx = agent.act(state)
                env.makeMove(action_map[action_idx])
                reward = env.reward()
                next_state = env.board.render_np().flatten()
                done = reward == 10 or reward == -10
                agent.memory.push(state, action_idx, reward, next_state, done)
                state = next_state
                total_reward += reward
                loss = agent.train_step(batch_size)
                if loss is not None:
                    loss_sum += loss; loss_count += 1
                if done: break
            if epoch % 10 == 0: agent.update_target_network()
            rewards_history.append(total_reward)
            loss_history.append(loss_sum / loss_count if loss_count > 0 else 0)
            if epoch % 20 == 0 or epoch == epochs - 1:
                progress_bar.progress((epoch + 1) / epochs)
                status_text.text(f"Epoch {epoch}/{epochs} | Reward: {total_reward} | (NoisyNet Auto-Exploration)")
                fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                ax[0].plot(rewards_history); ax[0].set_title("Total Reward")
                ax[1].plot(loss_history, color='orange'); ax[1].set_title("Average Loss")
                chart_empty.pyplot(fig); plt.close(fig)
        
    progress_bar.progress(1.0)
    status_text.text("Training Finished!")
    
    st.subheader("Test Run (Animation Loop)")
    st.markdown("Watching the agent play continuously. Adjust sidebar to stop.")
    
    board_placeholder = st.empty()
    status_placeholder = st.empty()
    
    if "HW3-4" in assignment_part:
        agent.q_network.eval() # Disable noise for evaluation
        
    # Infinite loop for animation
    while True:
        if "HW3-1" in assignment_part:
            env = Gridworld(size=4, mode='static')
        elif "HW3-2" in assignment_part:
            env = Gridworld(size=4, mode='player')
        else:
            env = Gridworld(size=4, mode='random')
            
        state = env.board.render_np().flatten()
        if hasattr(agent, 'epsilon'):
            agent.epsilon = 0.0 # pure greedy for evaluation
            
        test_steps = [(env.display(), "Start", 0)]
        for step in range(max_steps):
            action_idx = agent.act(state)
            action = action_map[action_idx]
            env.makeMove(action)
            reward = env.reward()
            done = reward == 10 or reward == -10
            test_steps.append((env.display(), action, reward))
            if done: break
            state = env.board.render_np().flatten()
            
        for i, (board_display, action, reward) in enumerate(test_steps):
            board_placeholder.markdown(render_board_html(board_display), unsafe_allow_html=True)
            status_info = f"**Step {i}**: Action: `{action}`, Reward: `{reward}`"
            if reward == 10:
                status_placeholder.success(status_info + "\n\n🎉 **Goal Reached!**")
            elif reward == -10:
                status_placeholder.error(status_info + "\n\n💥 **Fell into Pit!**")
            else:
                status_placeholder.info(status_info)
            time.sleep(0.5)
            
        time.sleep(2.0) # Wait 2 seconds before restarting the episode loop
