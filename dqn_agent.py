# dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class DQN(nn.Module):
    """
    Deep Q-Network architecture for LunarLander.

    Architecture:
    - Input: 8-dimensional state vector (x, y, vx, vy, angle, angular_velocity, left_leg_contact, right_leg_contact)
    - Hidden layers: 2 fully connected layers with ReLU activation
    - Output: 4 Q-values corresponding to 4 possible actions (do nothing, fire left engine, fire main engine, fire right engine)
    """

    def __init__(self, state_dim=8, action_dim=4, hidden_dim=128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """
    Experience Replay Buffer for storing and sampling transitions.

    Experience replay breaks the correlation between consecutive samples,
    improving training stability and sample efficiency.
    """

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent with experience replay and target network.

    Key components:
    1. Policy Network: Current Q-network used for action selection
    2. Target Network: Stabilizes training by providing fixed Q-targets
    3. Experience Replay: Breaks correlation between consecutive samples
    4. Epsilon-Greedy: Balances exploration vs exploitation
    """

    def __init__(
        self,
        state_dim=8,
        action_dim=4,
        hidden_dim=128,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=100000,
        batch_size=64,
        target_update_freq=10,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Hyperparameters
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0

        # Networks
        self.policy_net = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for stability

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state, training=True):
        """
        Epsilon-greedy action selection.

        With probability epsilon: choose random action (exploration)
        With probability 1-epsilon: choose best action from Q-network (exploitation)
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def update(self):
        """
        Perform one gradient descent step on a batch of transitions.

        DQN Update Rule:
        Loss = (r + γ * max_a' Q_target(s', a') - Q_policy(s, a))^2
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (using target network)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss and update
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()

        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Save model checkpoint."""
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            filepath,
        )

    def load(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
