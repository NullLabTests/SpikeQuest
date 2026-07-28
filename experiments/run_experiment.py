#!/usr/bin/env python3
"""Run a full multi-seed SpikeQuest experiment and plot results.

Usage:
    python experiments/run_experiment.py --config configs/experiment.yaml --seeds 5
"""

import argparse
import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from spikequest.env.grid_world import GridWorld
from spikequest.agents.spike_agent import SpikeAgent, train_episode
from spikequest.utils.seeding import set_seed
from spikequest.utils.logging import ExperimentLogger
from spikequest.utils.metrics import compute_metrics

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)


def make_agent(env: GridWorld, cfg: dict, device: str = "cpu") -> SpikeAgent:
    obs_dim = env.get_obs_dim()
    n_input = obs_dim * env.size if not env.partial_obs else obs_dim
    if not env.partial_obs:
        n_input = 2 * env.size  # population-coded (x, y)

    return SpikeAgent(
        n_input=n_input,
        n_hidden=cfg["agent"]["n_hidden"],
        n_output=cfg["agent"]["n_output"],
        T=cfg["agent"]["T"],
        novelty_coeff=cfg["agent"]["novelty_coeff"],
        baseline_decay=cfg["agent"]["baseline_decay"],
        rstdp_config=cfg.get("rstdp", {}),
        neuron_config=cfg.get("neuron", {}),
        novelty_mode=cfg["agent"]["novelty_mode"],
        device=device,
    )


class TabularQLearning:
    """Baseline: tabular Q-learning with epsilon-greedy exploration."""

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.95, epsilon: float = 0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def act(self, state: int) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        target = reward
        if not done:
            target += self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.lr * (target - self.Q[state, action])


def train_tabular_episode(env: GridWorld, agent: TabularQLearning,
                          max_steps: int = 200) -> dict:
    obs = env.reset()
    state_id = obs[0] * env.size + obs[1]
    total_reward = 0.0
    steps = 0
    success = False
    for _ in range(max_steps):
        action = agent.act(state_id)
        next_obs, reward, done, _ = env.step(action)
        next_id = next_obs[0] * env.size + next_obs[1]
        agent.update(state_id, action, reward, next_id, done)
        total_reward += reward
        state_id = next_id
        steps += 1
        if done:
            if next_obs[0] == env.goal_pos[0] and next_obs[1] == env.goal_pos[1]:
                success = True
            break
    return {"success": success, "total_reward": total_reward, "steps": steps}


def run_seed(cfg: dict, seed: int, device: str = "cpu") -> list:
    set_seed(seed)
    env = GridWorld(**cfg["env"])
    agent = make_agent(env, cfg, device)
    results = []
    for ep in range(cfg["experiment"]["n_episodes"]):
        r = train_episode(env, agent, max_steps=cfg["env"]["max_steps"])
        r["episode"] = ep
        results.append(r)
    return results


def run_tabular_seed(cfg: dict, seed: int) -> list:
    set_seed(seed)
    env = GridWorld(**cfg["env"])
    bcfg = cfg["baselines"]["tabular_q"]
    q_agent = TabularQLearning(env.size * env.size, 4,
                                lr=bcfg["lr"], gamma=bcfg["gamma"],
                                epsilon=bcfg["epsilon"])
    results = []
    for ep in range(cfg["experiment"]["n_episodes"]):
        r = train_tabular_episode(env, q_agent, max_steps=cfg["env"]["max_steps"])
        r["episode"] = ep
        results.append(r)
    return results


def plot_learning_curves(all_seeds, tabular_seeds=None, save_path="experiments/outputs/learning_curves.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for idx, (metric, ylabel, title) in enumerate([
        ("success", "Success Rate", "Success Rate"),
        ("total_reward", "Cumulative Reward", "Cumulative Reward"),
        ("steps", "Steps to Termination", "Steps"),
    ]):
        ax = axes[idx]
        all_seeds_array = np.array(all_seeds)
        if all_seeds_array.ndim == 3:
            data = all_seeds_array[:, :, idx]
        else:
            key_map = {"success": 0, "total_reward": 1, "steps": 2}
            data = np.zeros((len(all_seeds), len(all_seeds[0])))
            for s, seed_data in enumerate(all_seeds):
                for ep, ep_data in enumerate(seed_data):
                    data[s, ep] = ep_data[metric]

        mean = data.mean(axis=0)
        std = data.std(axis=0)
        episodes = np.arange(data.shape[1])

        ax.plot(episodes, mean, color="tab:blue", label="SNN (R-STDP)")
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.2, color="tab:blue")

        if tabular_seeds is not None:
            tab_data = np.array(tabular_seeds)
            t_mean = tab_data.mean(axis=0)
            t_std = tab_data.std(axis=0)
            ax.plot(episodes, t_mean, color="tab:orange", label="Tabular Q")
            ax.fill_between(episodes, t_mean - t_std, t_mean + t_std, alpha=0.2, color="tab:orange")

        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved learning curves to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    all_results = []
    for seed in range(args.seeds):
        print(f"Seed {seed}...")
        results = run_seed(cfg, seed, device)
        all_results.append(results)

    tabular_results = None
    if cfg["baselines"]["tabular_q"]["enabled"]:
        print("Running tabular Q-learning baseline...")
        tabular_results = []
        for seed in range(args.seeds):
            results = run_tabular_seed(cfg, seed)
            tabular_results.append(results)

    plot_learning_curves(all_results, tabular_results)

    # Final metrics
    final_ep = -10  # last 10 episodes
    snn_success = [np.mean([r["success"] for r in seed_res[final_ep:]]) for seed_res in all_results]
    print(f"SNN final success rate: {np.mean(snn_success):.2f} +/- {np.std(snn_success):.2f}")

    if tabular_results:
        tab_success = [np.mean([r["success"] for r in seed_res[final_ep:]]) for seed_res in tabular_results]
        print(f"Tabular Q final success rate: {np.mean(tab_success):.2f} +/- {np.std(tab_success):.2f}")

    print("Done.")


if __name__ == "__main__":
    main()