#!/usr/bin/env python3
"""Multi-seed comparison of SNN agent vs baselines on a 10x10 grid with obstacles.

Usage:
    python experiments/run_comparison.py --seeds 10 --episodes 300
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml

from spikequest.env.grid_world import GridWorld
from spikequest.agents.spike_agent import SpikeAgent, train_episode
from spikequest.utils.seeding import set_seed

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


class TabularQLearning:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.95, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target += self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.lr * (target - self.Q[state, action])


def make_snn_agent(env, cfg):
    return SpikeAgent(
        n_input=2 * env.size,
        n_hidden=cfg.get("n_hidden", 64),
        n_output=4,
        T=cfg.get("T", 10),
        novelty_coeff=cfg.get("novelty_coeff", 1.0),
        baseline_decay=cfg.get("baseline_decay", 0.9),
        rstdp_config=cfg.get("rstdp", {}),
        neuron_config=cfg.get("neuron", {}),
        novelty_mode=cfg.get("novelty_mode", "visited"),
        device="cpu",
    )


def run_snn_seed(env_cfg, agent_cfg, seed, n_episodes, max_steps):
    set_seed(seed)
    env = GridWorld(**env_cfg)
    agent = make_snn_agent(env, agent_cfg)
    successes, rewards, steps = [], [], []
    for ep in range(n_episodes):
        r = train_episode(env, agent, max_steps)
        successes.append(r["success"])
        rewards.append(r["total_reward"])
        steps.append(r["steps"])
    return successes, rewards, steps


def run_tabular_seed(env_cfg, seed, n_episodes, lr=0.1, gamma=0.95, epsilon=0.1):
    set_seed(seed)
    env = GridWorld(**env_cfg)
    q = TabularQLearning(env.size * env.size, 4, lr, gamma, epsilon)
    successes, rewards, steps = [], [], []
    for ep in range(n_episodes):
        obs = env.reset()
        s = int(obs[0] * env.size + obs[1])
        total_r = 0.0
        n_steps = 0
        done = False
        while not done and n_steps < env.max_steps:
            a = q.act(s)
            next_obs, r, done, _ = env.step(a)
            ns = int(next_obs[0] * env.size + next_obs[1])
            q.update(s, a, r, ns, done)
            total_r += r
            s = ns
            n_steps += 1
        successes.append(done and s == env.goal_pos[0] * env.size + env.goal_pos[1])
        rewards.append(total_r)
        steps.append(n_steps)
    return successes, rewards, steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--output", default="experiments/outputs/comparison")
    args = parser.parse_args()

    env_cfg = dict(size=10, max_steps=200, reward_goal=10.0, reward_step=0.0,
                   reward_obstacle=-0.1)

    snn_cfg = dict(
        n_hidden=64, T=10, novelty_coeff=1.0, baseline_decay=0.9,
        novelty_mode="visited",
        rstdp=dict(tau_pre=20.0, tau_post=20.0, tau_elig=50.0,
                   lr=0.005, w_init=0.3, w_init_std=0.1),
        neuron=dict(tau_m=20.0, V_th=1.0, V_reset=0.0),
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Running {args.seeds} seeds x {args.episodes} episodes...")
    print("SNN agent...")
    snn_all = [run_snn_seed(env_cfg, snn_cfg, s, args.episodes, 200)
               for s in range(args.seeds)]
    print("Tabular Q...")
    tab_all = [run_tabular_seed(env_cfg, s, args.episodes)
               for s in range(args.seeds)]

    snn_success = np.array([s[0] for s in snn_all], dtype=float)
    snn_rewards = np.array([s[1] for s in snn_all], dtype=float)
    snn_steps = np.array([s[2] for s in snn_all], dtype=float)

    tab_success = np.array([s[0] for s in tab_all], dtype=float)
    tab_rewards = np.array([s[1] for s in tab_all], dtype=float)
    tab_steps = np.array([s[2] for s in tab_all], dtype=float)

    # Smooth with running mean window=10
    def smooth(data, w=10):
        return np.array([np.convolve(d, np.ones(w)/w, mode="valid") for d in data])

    w = 10
    snn_ss = smooth(snn_success, w)
    tab_ss = smooth(tab_success, w)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, data_snn, data_tab, yl, tl in [
        (axes[0], snn_ss, tab_ss, "Success Rate", "Success Rate"),
        (axes[1], smooth(snn_rewards, w), smooth(tab_rewards, w), "Cumulative Reward", "Cumulative Reward"),
        (axes[2], smooth(snn_steps, w), smooth(tab_steps, w), "Steps", "Steps to Termination"),
    ]:
        ep = np.arange(data_snn.shape[1])
        m_snn = data_snn.mean(axis=0)
        s_snn = data_snn.std(axis=0)
        m_tab = data_tab.mean(axis=0)
        s_tab = data_tab.std(axis=0)

        ax.plot(ep, m_snn, label="SNN (R-STDP)", color="tab:blue")
        ax.fill_between(ep, m_snn - s_snn, m_snn + s_snn, alpha=0.2, color="tab:blue")
        ax.plot(ep, m_tab, label="Tabular Q", color="tab:orange")
        ax.fill_between(ep, m_tab - s_tab, m_tab + s_tab, alpha=0.2, color="tab:orange")
        ax.set_xlabel("Episode")
        ax.set_ylabel(yl)
        ax.set_title(tl)
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = out / "comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison to {path}")
    plt.close()

    # Print final metrics (last 20% of episodes)
    cutoff = int(args.episodes * 0.8)
    print("\n--- Final metrics (last 20% episodes) ---")
    print(f"SNN success rate:      {snn_success[:, cutoff:].mean():.3f} +/- {snn_success[:, cutoff:].std():.3f}")
    print(f"Tabular Q success:     {tab_success[:, cutoff:].mean():.3f} +/- {tab_success[:, cutoff:].std():.3f}")
    print(f"SNN reward:            {snn_rewards[:, cutoff:].mean():.2f} +/- {snn_rewards[:, cutoff:].std():.2f}")
    print(f"Tabular Q reward:      {tab_rewards[:, cutoff:].mean():.2f} +/- {tab_rewards[:, cutoff:].std():.2f}")


if __name__ == "__main__":
    main()