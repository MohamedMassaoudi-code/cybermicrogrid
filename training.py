# training.py
import time
import numpy as np
import pandas as pd
import torch
from .environment import PowerGridEnvironment

def train_value_based_agent(pp_net, data, agent, num_episodes=50, attack_mode="none", attack_strength=0.1):
    env = PowerGridEnvironment(pp_net, data, attack_mode=attack_mode, attack_strength=attack_strength)
    action_dim = len(env.action_space)
    rewards_history = []
    steps_history = []
    iter_times = []
    freq_history = []
    power_history = []
    for ep in range(num_episodes):
        ep_start = time.time()
        st = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        while not done and steps < 20:
            act = agent.select_action(st, action_dim)
            ns, rew, done = env.step(act)
            agent.store_transition(st, act, rew, ns, done)
            st = ns
            total_reward += rew
            steps += 1
            if hasattr(agent, "optimize_model"):
                agent.optimize_model()
        ep_time = time.time() - ep_start
        rewards_history.append(total_reward)
        steps_history.append(steps)
        iter_times.append(ep_time)
        freq_history.append(env.compute_frequency())
        power_history.append(env.compute_active_power())
        if hasattr(agent, "target_update") and ep % agent.target_update == 0:
            agent.update_target_network()
        print(f"[{attack_mode.upper()}][Episode {ep+1:03d}/{num_episodes}] Reward: {total_reward:.2f}, Steps: {steps}, Time: {ep_time:.3f}s")
    return env, rewards_history, freq_history, power_history, steps_history, iter_times

def train_policy_gradient_agent(pp_net, data, agent, num_episodes=50, attack_mode="none", attack_strength=0.1):
    env = PowerGridEnvironment(pp_net, data, attack_mode=attack_mode, attack_strength=attack_strength)
    action_dim = len(env.action_space)
    rewards_history = []
    steps_history = []
    iter_times = []
    freq_history = []
    power_history = []
    for ep in range(num_episodes):
        ep_start = time.time()
        st = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        while not done and steps < 20:
            act = agent.select_action(st, action_dim)
            ns, rew, done = env.step(act)
            agent.store_transition(st, act, rew, ns, done)
            st = ns
            total_reward += rew
            steps += 1
        agent.update_policy()
        ep_time = time.time() - ep_start
        rewards_history.append(total_reward)
        steps_history.append(steps)
        iter_times.append(ep_time)
        freq_history.append(env.compute_frequency())
        power_history.append(env.compute_active_power())
        print(f"[{attack_mode.upper()}][Episode {ep+1:03d}/{num_episodes}] Reward: {total_reward:.2f}, Steps: {steps}, Time: {ep_time:.3f}s")
    return env, rewards_history, freq_history, power_history, steps_history, iter_times

def ultra_fast_training(pp_net, data, agent, num_episodes=50, attack_mode="none", attack_strength=0.1):
    env = PowerGridEnvironment(pp_net, data, attack_mode=attack_mode, attack_strength=attack_strength)
    agent.set_attack_mode(attack_mode)
    rewards_history = []
    freq_history = []
    power_history = []
    feasible_history = []
    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        while not done and steps < 20:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_outcome(reward, done)
            state = next_state
            total_reward += reward
            steps += 1
        feasible_history.append(1 if env.feasible else 0)
        if done:
            next_value = 0.0
        else:
            if not hasattr(state, "batch"):
                state.batch = torch.zeros(state.num_nodes, dtype=torch.long)
            with torch.no_grad():
                _, value = agent.policy(state)
                next_value = value.view(-1).item()
        agent.update()
        rewards_history.append(total_reward)
        freq_history.append(env.compute_frequency())
        power_history.append(env.compute_active_power())
        print(f"[TGAP][{attack_mode.upper()}][Episode {ep+1}/{num_episodes}] Reward={total_reward:.2f}")
    return env, rewards_history, freq_history, power_history, feasible_history
