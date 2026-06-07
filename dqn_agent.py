"""
Deep Q-Network (DQN) for Chess
Implements:
  - Convolutional DQN with residual blocks
  - Experience Replay Buffer
  - Target Network (hard update)
  - Legal-move masking during action selection
"""

import numpy as np
import random
from collections import deque
from typing import Tuple, List

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not found. Install with: pip install torch")


# ──────────────────────────────────────────────────────────────────────────────
# Replay Buffer
# ──────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Fixed-size circular buffer storing (s, a, r, s', done, mask') tuples."""

    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_mask):
        self.buffer.append((state, action, reward, next_state, done, next_mask))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.float32),
                np.array(next_masks))

    def __len__(self):
        return len(self.buffer)


# ──────────────────────────────────────────────────────────────────────────────
# Network Architecture
# ──────────────────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:

    class ResBlock(nn.Module):
        """Residual block used in AlphaZero-style networks."""
        def __init__(self, channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn1   = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn2   = nn.BatchNorm2d(channels)

        def forward(self, x):
            residual = x
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            return F.relu(x + residual)


    class ChessDQN(nn.Module):
        """
        Input:  (B, 14, 8, 8)  — board state planes
        Output: (B, 4096)      — Q-value for every (from, to) action
        """

        def __init__(self,
                     in_channels: int = 14,
                     filters: int = 128,
                     num_res_blocks: int = 6,
                     action_size: int = 4096):
            super().__init__()

            # Entry convolution
            self.entry = nn.Sequential(
                nn.Conv2d(in_channels, filters, 3, padding=1, bias=False),
                nn.BatchNorm2d(filters),
                nn.ReLU()
            )

            # Residual tower
            self.res_tower = nn.Sequential(
                *[ResBlock(filters) for _ in range(num_res_blocks)]
            )

            # Value head (flattened)
            self.value_head = nn.Sequential(
                nn.Conv2d(filters, 32, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32 * 8 * 8, 256),
                nn.ReLU(),
                nn.Linear(256, action_size)
            )

        def forward(self, x):
            # x: (B, 8, 8, 14) → transpose to (B, 14, 8, 8)
            x = x.permute(0, 3, 1, 2).float()
            x = self.entry(x)
            x = self.res_tower(x)
            return self.value_head(x)   # (B, 4096)


# ──────────────────────────────────────────────────────────────────────────────
# DQN Agent
# ──────────────────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    DQN agent with:
      - ε-greedy exploration with legal-move masking
      - Experience replay
      - Hard target-network updates every `target_update` steps
    """

    def __init__(self,
                 state_shape: Tuple,
                 action_size: int = 4096,
                 lr: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.9995,
                 batch_size: int = 64,
                 buffer_size: int = 50_000,
                 target_update: int = 500,
                 device: str = "auto"):

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required. pip install torch")

        self.action_size   = action_size
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.steps         = 0

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Networks
        self.policy_net = ChessDQN(action_size=action_size).to(self.device)
        self.target_net = ChessDQN(action_size=action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory    = ReplayBuffer(buffer_size)

    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray, legal_mask: np.ndarray) -> int:
        """ε-greedy with illegal-move masking."""
        legal_indices = np.where(legal_mask)[0]

        if random.random() < self.epsilon:
            return int(random.choice(legal_indices))

        state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t).squeeze(0).cpu().numpy()

        # Mask illegal moves with -inf
        masked = np.full(self.action_size, -np.inf)
        masked[legal_indices] = q_values[legal_indices]
        return int(np.argmax(masked))

    # ------------------------------------------------------------------
    def store(self, state, action, reward, next_state, done, next_mask):
        self.memory.push(state, action, reward, next_state, done, next_mask)

    # ------------------------------------------------------------------
    def train_step(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, next_masks = \
            self.memory.sample(self.batch_size)

        states_t      = torch.from_numpy(states).to(self.device)
        actions_t     = torch.from_numpy(actions).long().to(self.device)
        rewards_t     = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t       = torch.from_numpy(dones).to(self.device)
        next_masks_t  = torch.from_numpy(next_masks).to(self.device)

        # Current Q
        q_current = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Next Q from target network with masking
        with torch.no_grad():
            q_next_all = self.target_net(next_states_t)
            q_next_all[~next_masks_t] = -1e9   # mask illegal
            q_next = q_next_all.max(1).values

        q_target = rewards_t + self.gamma * q_next * (1 - dones_t)

        loss = F.smooth_l1_loss(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Hard target update
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "epsilon":    self.epsilon,
            "steps":      self.steps,
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["policy_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.steps   = ckpt["steps"]
        print(f"Model loaded from {path}")
