#!/usr/bin/env python3
"""Trains a SpikeQuest agent on 10x10 grid with obstacles and visualises results.

This notebook-style script walks through a full experiment:
  1. Environment setup
  2. Agent initialisation with R-STDP
  3. Training loop with logging
  4. Weight evolution visualisation
  5. Path animation
  6. Learning curves
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

from spikequest.env.grid_world import GridWorld
from spikequest.agents.spike_agent import SpikeAgent, train_episode
from spikequest.utils.seeding import set_seed

set_seed(42)

env = GridWorld(size=10, max_steps=200)
agent = SpikeAgent(
    n_input=2 * env.size,
    n_hidden=64,
    n_output=4,
    T=10,
    novelty_coeff=1.0,
    baseline_decay=0.9,
    rstdp_config=dict(tau_pre=20.0, tau_post=20.0, tau_elig=50.0,
                       lr=0.005, w_init=0.3, w_init_std=0.1),
    novelty_mode="visited",
)

n_episodes = 150
all_success = []
all_rewards = []
all_steps = []

for ep in range(n_episodes):
    r = train_episode(env, agent, max_steps=200)
    all_success.append(r["success"])
    all_rewards.append(r["total_reward"])
    all_steps.append(r["steps"])
    if (ep + 1) % 25 == 0:
        sr = np.mean(all_success[-25:])
        print(f"Episode {ep+1}/{n_episodes}, success rate (last 25): {sr:.2f}")

# Learning curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
window = 10

def smooth(x, w):
    return np.convolve(x, np.ones(w)/w, mode='valid')

axes[0].plot(smooth(all_success, window), color='tab:blue')
axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Success Rate')
axes[0].set_title(f'Success Rate (window={window})')

axes[1].plot(smooth(all_rewards, window), color='tab:green')
axes[1].set_xlabel('Episode'); axes[1].set_ylabel('Cumulative Reward')
axes[1].set_title(f'Cumulative Reward (window={window})')

axes[2].plot(smooth(all_steps, window), color='tab:red')
axes[2].set_xlabel('Episode'); axes[2].set_ylabel('Steps')
axes[2].set_title(f'Steps to Termination (window={window})')

plt.tight_layout()
out_dir = Path("experiments/outputs/demo")
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / "learning_curves.png", dpi=150, bbox_inches='tight')
plt.close()

# Final evaluation
eval_agent = SpikeAgent(
    n_input=2 * env.size, n_hidden=64, n_output=4, T=10,
    novelty_coeff=0.0, rstdp_config=dict(lr=0.0),
    novelty_mode="visited",
)
eval_agent.set_weights(agent.policy.get_weights())

env_eval = GridWorld(size=10, max_steps=50)
obs = env_eval.reset()
path = [(0, 0)]

for _ in range(50):
    action = eval_agent.act(obs, env_eval.size)
    obs, reward, done, info = env_eval.step(action)
    path.append(env_eval.agent_pos)
    if done:
        break

print(f"Evaluation path length: {len(path)}")
print(f"Goal reached: {path[-1] == env_eval.goal_pos}")

# Render grid
print("\nGrid path (digits = step order, # = obstacle, G = goal):")
print(env_eval.render_grid(path))

print("\nDone. See experiments/outputs/demo/ for visualisations.")