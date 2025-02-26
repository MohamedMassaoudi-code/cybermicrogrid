# agents.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Batch

from .utils import state_to_tensor

# Dummy Agent
class DummyAgent:
    def __init__(self, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.05):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def select_action(self, state, action_dim):
        return random.randrange(action_dim)

    def store_transition(self, s, a, r, ns, d):
        pass

    def optimize_model(self):
        pass

    def update_target_network(self):
        pass

# Q-Network for DQN
class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Replay Memory for DQN
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# DQNAgent
class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, batch_size=32,
                 gamma=0.99, lr=1e-3, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05,
                 target_update=10, memory_capacity=10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update = target_update

        self.policy_net = QNetwork(state_dim, hidden_dim, action_dim)
        self.target_net = QNetwork(state_dim, hidden_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_capacity)
        self.steps_done = 0

    def select_action(self, state, action_dim):
        if random.random() < self.epsilon:
            return random.randrange(action_dim)
        else:
            with torch.no_grad():
                st_tensor = state_to_tensor(state).float().unsqueeze(0)
                q_vals = self.policy_net(st_tensor)
                return q_vals.max(1)[1].item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(
            (state_to_tensor(state).float(),
             action,
             torch.tensor([reward], dtype=torch.float),
             state_to_tensor(next_state).float(),
             torch.tensor([done], dtype=torch.float))
        )

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        bs, ba, br, bns, bd = zip(*transitions)
        bs = torch.stack(bs)
        ba = torch.tensor(ba).unsqueeze(1)
        br = torch.cat(br)
        bns = torch.stack(bns)
        bd = torch.cat(bd)
        curr_q = self.policy_net(bs).gather(1, ba)
        with torch.no_grad():
            max_next_q = self.target_net(bns).max(1)[0]
            exp_q = br + (1 - bd) * self.gamma * max_next_q
        exp_q = exp_q.unsqueeze(1)
        loss = nn.MSELoss()(curr_q, exp_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# DoubleDQNAgent (DDQN)
class DoubleDQNAgent(DQNAgent):
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        bs, ba, br, bns, bd = zip(*transitions)
        bs = torch.stack(bs)
        ba = torch.tensor(ba).unsqueeze(1)
        br = torch.cat(br)
        bns = torch.stack(bns)
        bd = torch.cat(bd)
        curr_q = self.policy_net(bs).gather(1, ba)
        with torch.no_grad():
            next_actions = self.policy_net(bns).max(1)[1].unsqueeze(1)
            next_q = self.target_net(bns).gather(1, next_actions).squeeze(1)
            exp_q = br + (1 - bd) * self.gamma * next_q
        exp_q = exp_q.unsqueeze(1)
        loss = nn.MSELoss()(curr_q, exp_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Dueling Q-Network
class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.value_fc = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        self.adv_fc = nn.Linear(hidden_dim, hidden_dim)
        self.advantage = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        val = self.value_fc(x)
        val = self.relu(val)
        val = self.value(val)
        adv = self.adv_fc(x)
        adv = self.relu(adv)
        adv = self.advantage(adv)
        return val + (adv - adv.mean(dim=1, keepdim=True))

# DuelingDQNAgent
class DuelingDQNAgent(DQNAgent):
    def __init__(self, state_dim, action_dim, hidden_dim=64, batch_size=32,
                 gamma=0.99, lr=1e-3, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05,
                 target_update=10, memory_capacity=10000):
        super().__init__(state_dim, action_dim, hidden_dim, batch_size,
                         gamma, lr, epsilon, epsilon_decay, epsilon_min,
                         target_update, memory_capacity)
        self.policy_net = DuelingQNetwork(state_dim, hidden_dim, action_dim)
        self.target_net = DuelingQNetwork(state_dim, hidden_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

# Policy Network for Policy Gradient
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=-1)

# Policy Gradient Agent
class PolicyGradientAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=1e-3, gamma=0.99):
        self.policy_net = PolicyNetwork(state_dim, hidden_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []

    def select_action(self, state, action_dim):
        st = state_to_tensor(state).float().unsqueeze(0)
        probs = self.policy_net(st)
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def store_transition(self, s, a, r, ns, d):
        self.rewards.append(r)

    def update_policy(self):
        R = 0
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        loss = 0
        for logp, Gt in zip(self.log_probs, returns):
            loss -= logp * Gt
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.log_probs = []
        self.rewards = []

# LightningFastPPO Network
from torch_geometric.nn import GCNConv, global_mean_pool

class LightningFastPPO(nn.Module):
    """
    Ultra-lightweight graph neural network for power grid defense.
    """
    def __init__(self, in_channels, hidden_dim, action_dim):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.skip = nn.Linear(in_channels, hidden_dim) if in_channels != hidden_dim else nn.Identity()
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.activation = F.relu
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.4)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        identity = self.skip(x)
        x = self.conv1(x, edge_index)
        x = self.activation(x + identity)
        identity = x
        x = self.conv2(x, edge_index)
        x = self.activation(x + identity)
        if hasattr(data, "batch") and data.batch is not None:
            x = global_mean_pool(x, data.batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        action_logits = self.actor(x)
        state_values = self.critic(x)
        return action_logits, state_values

# UltraFastPPOAgent (TGAP)
class UltraFastPPOAgent:
    def __init__(self, in_channels, action_dim, hidden_dim=64,
                 gamma=0.99, eps_clip=0.1, lam=0.9,
                 K_epochs=3, mini_batch_size=16, lr=5e-4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.lam = lam
        self.K_epochs = K_epochs
        self.mini_batch_size = mini_batch_size
        self.policy = LightningFastPPO(in_channels, hidden_dim, action_dim)
        self.optimizer = optim.SGD(self.policy.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
        self.lr_scheduler = lambda epoch: max(0.3, 1.0 - 0.03 * epoch)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_scheduler)
        self.reset_buffers()
        self.running_reward = 0
        self.best_reward = float('-inf')
        self.temperature = 1.0
        self.min_temperature = 0.3
        self.temp_decay = 0.9
        self.target_kl = 0.015
        self.attack_mode = "none"

    def reset_buffers(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def set_attack_mode(self, mode):
        self.attack_mode = mode.upper()
        if self.attack_mode in ["DOS", "DDOS"]:
            self.temperature = 0.5
            self.min_temperature = 0.1
            self.eps_clip = 0.05
        elif self.attack_mode in ["RANSOMWARE", "ZERO_DAY"]:
            self.temperature = 1.5
            self.min_temperature = 0.4
            self.eps_clip = 0.2
        else:
            self.temperature = 1.0
            self.min_temperature = 0.3
            self.eps_clip = 0.1

    def select_action(self, state_data):
        if not hasattr(state_data, "batch") or state_data.batch is None:
            state_data.batch = torch.zeros(state_data.num_nodes, dtype=torch.long)
        self.policy.eval()
        with torch.no_grad():
            logits, value = self.policy(state_data)
            logits = logits.squeeze(0)
            scaled_logits = logits / max(0.1, self.temperature)
            probs = F.softmax(scaled_logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
            log_prob = torch.log(probs[action] + 1e-10)
        self.states.append(state_data.clone())
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        return action

    def store_outcome(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)
        if done:
            episode_reward = sum(self.rewards)
            self.running_reward = 0.05 * episode_reward + 0.95 * self.running_reward
            improvement = (episode_reward - self.running_reward) / (abs(self.running_reward) + 1e-8)
            decay_rate = self.temp_decay * (1.0 - 0.5 * min(max(improvement, -1.0), 1.0))
            self.temperature = max(self.min_temperature, self.temperature * decay_rate)
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward

    def compute_advantages(self, next_value=0.0):
        rewards = torch.tensor(self.rewards, dtype=torch.float)
        dones = torch.tensor(self.dones, dtype=torch.float)
        values = torch.cat(self.values).squeeze(-1)
        next_value_tensor = torch.tensor([next_value], dtype=torch.float)
        values_extended = torch.cat([values, next_value_tensor])
        advantages = torch.zeros_like(rewards)
        gae = 0
        for i in reversed(range(len(rewards))):
            next_val = values_extended[i+1]
            delta = rewards[i] + self.gamma * next_val * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages[i] = gae
        returns = advantages + values
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns, values

    def update(self):
        if len(self.rewards) == 0:
            return 0.0
        next_value = 0.0
        if self.states and not self.dones[-1]:
            state = self.states[-1]
            if not hasattr(state, "batch") or state.batch is None:
                state.batch = torch.zeros(state.num_nodes, dtype=torch.long)
            with torch.no_grad():
                _, value = self.policy(state)
                next_value = value.item()
        advantages, returns, old_values = self.compute_advantages(next_value)
        batch_states = Batch.from_data_list(self.states)
        actions = torch.tensor(self.actions)
        old_log_probs = torch.stack(self.log_probs)
        dataset_size = len(self.rewards)
        indices = np.arange(dataset_size)
        kl_divs = []
        self.policy.train()
        for epoch in range(self.K_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.mini_batch_size):
                end = min(start + self.mini_batch_size, dataset_size)
                if end - start < 3:
                    continue
                mb_idx = indices[start:end]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_old_values = old_values[mb_idx]
                self.optimizer.zero_grad()
                logits, values = self.policy(batch_states)
                logits = logits.view(-1)
                values = values.view(-1)
                mini_logits = logits[mb_idx]
                mini_value = values[mb_idx]
                dist = Categorical(logits=mini_logits)
                new_log_probs = dist.log_prob(mb_actions)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_clipped = mb_old_values + torch.clamp(mini_value - mb_old_values, -self.eps_clip, self.eps_clip)
                value_loss_unclipped = F.mse_loss(mini_value, mb_returns)
                value_loss_clipped = F.mse_loss(value_clipped, mb_returns)
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
                entropy_loss = -0.01 * dist.entropy().mean()
                total_loss = policy_loss + 0.5 * value_loss + entropy_loss
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
                with torch.no_grad():
                    log_ratio = new_log_probs - mb_old_log_probs
                    kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                    kl_divs.append(kl)
                    if kl > self.target_kl * 2:
                        break
            if kl_divs and np.mean(kl_divs) > self.target_kl:
                break
        self.scheduler.step()
        self.reset_buffers()
        return np.mean(kl_divs) if kl_divs else 0.0
