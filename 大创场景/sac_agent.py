"""
sac_agent.py — Soft Actor-Critic (SAC) implementation for VSL control.

Components:
  - Actor (policy) network with reparameterized Gaussian sampling
  - Twin Critic (Q) networks with soft target updates
  - Replay buffer with uniform sampling
  - Automatic entropy tuning (learnable alpha)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam


# ================================================================
# Replay Buffer
# ================================================================

class ReplayBuffer:
    """
    Fixed-size circular replay buffer storing (s, a, r, s', done) transitions.
    Uses numpy arrays for memory efficiency.
    """

    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state      = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action     = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward     = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done       = np.zeros((max_size, 1), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        """Store one transition."""
        idx = self.ptr
        self.state[idx]      = state
        self.action[idx]     = action
        self.reward[idx]     = reward
        self.next_state[idx] = next_state
        self.done[idx]       = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """Randomly sample a batch of transitions as torch tensors."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.state[indices]),
            torch.from_numpy(self.action[indices]),
            torch.from_numpy(self.reward[indices]),
            torch.from_numpy(self.next_state[indices]),
            torch.from_numpy(self.done[indices]),
        )

    def __len__(self):
        return self.size


# ================================================================
# Neural Networks
# ================================================================

def _make_mlp(input_dim, hidden_dims, output_dim, use_layer_norm=True):
    """Build a sequential MLP with optional LayerNorm."""
    layers = []
    in_dim = input_dim
    for i, h_dim in enumerate(hidden_dims):
        layers.append(nn.Linear(in_dim, h_dim))
        if use_layer_norm and i < len(hidden_dims) - 1:
            layers.append(nn.LayerNorm(h_dim))
        layers.append(nn.ReLU())
        in_dim = h_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """
    Stochastic policy network.
    Outputs mean (tanh-bounded to [-1,1]) and log_std for each action dimension.
    """

    def __init__(self, state_dim, action_dim, hidden_sizes):
        super().__init__()
        self.action_dim = action_dim

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.LayerNorm(hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.LayerNorm(hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
        )
        self.mean_head     = nn.Linear(hidden_sizes[2], action_dim)
        self.log_std_head  = nn.Linear(hidden_sizes[2], action_dim)

        # Small constant for numerical stability
        self._eps = 1e-6

    def forward(self, state):
        """Return action mean and log_std for the given state."""
        x = self.backbone(state)
        mean = torch.tanh(self.mean_head(x))          # ∈ (-1, 1)
        log_std = self.log_std_head(x).clamp(-20, 2)  # bounded for stability
        return mean, log_std

    def sample(self, state):
        """
        Sample action using reparameterization trick.
        Returns: (action ∈ [-1,1], log_prob, mean)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        # Reparameterized sample
        u = normal.rsample()
        action = torch.tanh(u)

        # Log probability with tanh squashing correction
        # log π(a|s) = log μ(u|s) - sum(log(1 - tanh²(u) + ε))
        log_prob = normal.log_prob(u) - torch.log(1.0 - action.pow(2) + self._eps)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, mean

    def get_deterministic_action(self, state):
        """Return the mean action (no noise) for evaluation."""
        mean, _ = self.forward(state)
        return torch.tanh(mean)  # tanh just in case mean head output exceeds [-1,1]


class Critic(nn.Module):
    """
    Q-function network. Takes (state, action) concatenated, outputs scalar Q.
    """

    def __init__(self, state_dim, action_dim, hidden_sizes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_sizes[0]),
            nn.LayerNorm(hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.LayerNorm(hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], 1),
        )

    def forward(self, state, action):
        """Return Q(s, a) scalar."""
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


# ================================================================
# SAC Agent
# ================================================================

class SACAgent:
    """
    Soft Actor-Critic agent with automatic entropy tuning.

    Features:
      - Twin Q-networks to reduce overestimation
      - Soft target updates (Polyak averaging)
      - Reparameterized Gaussian policy
      - Learnable entropy coefficient alpha
    """

    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.actor = Actor(state_dim, action_dim, config.HIDDEN_SIZES).to(self.device)
        self.critic1 = Critic(state_dim, action_dim, config.HIDDEN_SIZES).to(self.device)
        self.critic2 = Critic(state_dim, action_dim, config.HIDDEN_SIZES).to(self.device)

        # Target networks (initialized with same weights)
        self.critic1_target = Critic(state_dim, action_dim, config.HIDDEN_SIZES).to(self.device)
        self.critic2_target = Critic(state_dim, action_dim, config.HIDDEN_SIZES).to(self.device)
        self._hard_update(self.critic1_target, self.critic1)
        self._hard_update(self.critic2_target, self.critic2)

        # Optimizers
        self.actor_optimizer    = Adam(self.actor.parameters(), lr=config.ACTOR_LR)
        self.critic1_optimizer  = Adam(self.critic1.parameters(), lr=config.CRITIC_LR)
        self.critic2_optimizer  = Adam(self.critic2.parameters(), lr=config.CRITIC_LR)

        # Entropy coefficient (learnable)
        self.log_alpha = torch.tensor(config.INITIAL_LOG_ALPHA, requires_grad=True, device=self.device)
        self.alpha_optimizer = Adam([self.log_alpha], lr=config.ALPHA_LR)
        self.target_entropy = config.TARGET_ENTROPY

        # Training stats
        self.actor_loss = 0.0
        self.critic_loss = 0.0
        self.alpha_loss = 0.0
        self.alpha_value = float(self.log_alpha.exp().item())

    @property
    def alpha(self):
        return self.log_alpha.exp()

    # ---------------------------------------------------------------
    # Action Selection
    # ---------------------------------------------------------------

    def select_action(self, state, evaluate=False):
        """
        Select action given current state.

        Args:
            state: np.array (state_dim,)
            evaluate: if True, return deterministic action (mean)

        Returns:
            action: np.array (action_dim,) — values in [-1, 1]
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if evaluate:
                action = self.actor.get_deterministic_action(state_tensor)
            else:
                action, _, _ = self.actor.sample(state_tensor)

        return action.cpu().numpy().squeeze(0)

    # ---------------------------------------------------------------
    # Update
    # ---------------------------------------------------------------

    def update(self, replay_buffer):
        """
        Perform one SAC update step using a batch from the replay buffer.

        Returns:
            dict with loss values for logging
        """
        # Sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(self.config.BATCH_SIZE)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # ---- Update Critics ----
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            q1_target_next = self.critic1_target(next_states, next_actions)
            q2_target_next = self.critic2_target(next_states, next_actions)
            q_target_next = torch.min(q1_target_next, q2_target_next)
            # Bellman backup with entropy bonus
            q_target = rewards + self.config.GAMMA * (1.0 - dones) * (
                q_target_next - self.alpha * next_log_probs
            )

        # Current Q estimates
        q1_current = self.critic1(states, actions)
        q2_current = self.critic2(states, actions)

        # MSE loss
        critic1_loss = F.mse_loss(q1_current, q_target)
        critic2_loss = F.mse_loss(q2_current, q_target)
        critic_loss = critic1_loss + critic2_loss

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), self.config.GRADIENT_CLIP)
        nn.utils.clip_grad_norm_(self.critic2.parameters(), self.config.GRADIENT_CLIP)
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # ---- Update Actor ----
        new_actions, log_probs, _ = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.GRADIENT_CLIP)
        self.actor_optimizer.step()

        # ---- Update Alpha ----
        alpha_loss = -(self.alpha * (log_probs.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ---- Soft Update Target Networks ----
        self._soft_update(self.critic1_target, self.critic1)
        self._soft_update(self.critic2_target, self.critic2)

        # Store stats
        self.actor_loss  = float(actor_loss.item())
        self.critic_loss = float(critic_loss.item())
        self.alpha_loss  = float(alpha_loss.item())
        self.alpha_value = float(self.alpha.item())

        return {
            "actor_loss":  self.actor_loss,
            "critic_loss": self.critic_loss,
            "alpha_loss":  self.alpha_loss,
            "alpha":       self.alpha_value,
        }

    # ---------------------------------------------------------------
    # Target Network Updates
    # ---------------------------------------------------------------

    def _soft_update(self, target, source):
        """Polyak averaging: θ_target = τ * θ_source + (1-τ) * θ_target"""
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.config.TAU * sp.data + (1.0 - self.config.TAU) * tp.data)

    @staticmethod
    def _hard_update(target, source):
        """Copy source weights to target."""
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(sp.data)

    # ---------------------------------------------------------------
    # Save / Load
    # ---------------------------------------------------------------

    def save(self, filepath):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic1_optimizer": self.critic1_optimizer.state_dict(),
            "critic2_optimizer": self.critic2_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.critic1_target.load_state_dict(checkpoint["critic1_target"])
        self.critic2_target.load_state_dict(checkpoint["critic2_target"])
        self.log_alpha = torch.tensor(checkpoint["log_alpha"], requires_grad=True, device=self.device)
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic1_optimizer.load_state_dict(checkpoint["critic1_optimizer"])
        self.critic2_optimizer.load_state_dict(checkpoint["critic2_optimizer"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])


# ================================================================
# Simple test (run directly to verify)
# ================================================================

if __name__ == "__main__":
    print("Testing SAC agent components...")

    # Dummy config
    class DummyConfig:
        HIDDEN_SIZES = [256, 256, 128]
        ACTOR_LR = 3e-4
        CRITIC_LR = 3e-4
        ALPHA_LR = 3e-4
        GAMMA = 0.99
        TAU = 0.005
        BATCH_SIZE = 256
        GRADIENT_CLIP = 1.0
        INITIAL_LOG_ALPHA = -2.3026
        TARGET_ENTROPY = -1.0

    config = DummyConfig()
    state_dim, action_dim = 72, 1

    # Test Actor
    actor = Actor(state_dim, action_dim, config.HIDDEN_SIZES)
    s = torch.randn(4, state_dim)
    action, log_prob, mean = actor.sample(s)
    print(f"Actor: action shape={action.shape}, log_prob shape={log_prob.shape}, mean shape={mean.shape}")
    assert action.shape == (4, action_dim)
    assert log_prob.shape == (4, 1)

    # Test Critic
    critic = Critic(state_dim, action_dim, config.HIDDEN_SIZES)
    q = critic(s, action)
    print(f"Critic: Q shape={q.shape}")
    assert q.shape == (4, 1)

    # Test ReplayBuffer
    buf = ReplayBuffer(state_dim, action_dim, max_size=10000)
    for i in range(500):
        buf.push(np.random.randn(state_dim).astype(np.float32),
                 np.random.randn(action_dim).astype(np.float32),
                 np.array([np.random.randn()], dtype=np.float32),
                 np.random.randn(state_dim).astype(np.float32),
                 np.array([0.0], dtype=np.float32))
    ss, aa, rr, nss, dd = buf.sample(32)
    print(f"Buffer: len={len(buf)}, sample shapes: {ss.shape}, {aa.shape}, {rr.shape}")

    # Test SACAgent update
    agent = SACAgent(state_dim, action_dim, config)
    stats = agent.update(buf)
    print(f"Agent update: {stats}")

    # Test action selection
    a_train = agent.select_action(np.random.randn(state_dim).astype(np.float32), evaluate=False)
    a_eval  = agent.select_action(np.random.randn(state_dim).astype(np.float32), evaluate=True)
    print(f"Action (train): {a_train}, Action (eval): {a_eval}")
    print(f"Action shape: {a_train.shape}, range: [{a_train.min():.3f}, {a_train.max():.3f}]")

    print("\nAll tests passed!")
