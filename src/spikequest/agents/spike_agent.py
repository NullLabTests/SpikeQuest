import torch
import numpy as np
import random
from typing import Optional, List, Tuple, Dict

from ..env.grid_world import GridWorld
from ..networks.snn_policy import SNNPolicy
from ..novelty.novelty import VisitedNovelty, DualTimescaleNovelty
from ..learning.rstdp import RSTDP


class SpikeAgent:
    """SNN-based agent combining policy, learning, and novelty-driven exploration.

    At each environment step:
      1. Encode observation as a rate-coded spike pattern
      2. Run SNN for T simulation timesteps, accumulating eligibility traces
      3. Select action via output spike-count argmax
      4. Execute action in environment
      5. Compute novelty bonus + modulated reward
      6. Apply R-STDP weight update using the modulator

    Args:
        n_input: input encoding dimension
        n_hidden: hidden layer size(s)
        n_output: number of actions
        T: simulation timesteps per env step
        novelty_coeff: scaling factor for novelty bonus
        baseline_decay: exponential moving average decay for reward baseline
        device: torch device string
    """

    def __init__(
        self,
        n_input: int = 20,
        n_hidden: int = 64,
        n_output: int = 4,
        T: int = 10,
        novelty_coeff: float = 1.0,
        baseline_decay: float = 0.9,
        rstdp_config: Optional[dict] = None,
        neuron_config: Optional[dict] = None,
        novelty_mode: str = "visited",
        device: str = "cpu",
    ):
        self.policy = SNNPolicy(
            n_input=n_input,
            n_hidden=n_hidden,
            n_output=n_output,
            T=T,
            neuron_config=neuron_config,
            rstdp_config=rstdp_config,
            device=device,
        )
        self.n_output = n_output
        self.T = T
        self.novelty_coeff = novelty_coeff
        self.baseline = 0.0
        self.baseline_decay = baseline_decay
        self.device = device

        if novelty_mode == "visited":
            self.novelty_module = VisitedNovelty()
        elif novelty_mode == "dual":
            self.novelty_module = DualTimescaleNovelty()
        else:
            self.novelty_module = VisitedNovelty()

    def encode(self, obs: np.ndarray, grid_size: int) -> torch.Tensor:
        """Population-code encoding of (x, y) position."""
        if len(obs) == 2:
            x, y = int(obs[0]), int(obs[1])
            encoding = torch.zeros(2 * grid_size, device=self.device)
            encoding[x] = 1.0
            encoding[grid_size + y] = 1.0
        else:
            encoding = torch.from_numpy(obs).float().to(self.device)
            if encoding.dim() == 0:
                encoding = encoding.unsqueeze(0)
            while encoding.dim() < 1:
                encoding = encoding.unsqueeze(0)
            need = 2 * grid_size - encoding.shape[0]
            if need > 0:
                encoding = torch.cat([encoding, torch.zeros(need, device=self.device)])
        return encoding

    def act(self, obs: np.ndarray, grid_size: int) -> int:
        encoded = self.encode(obs, grid_size)
        self.policy.reset_state()
        output_counts = self.policy.forward_sim(encoded)
        if output_counts.sum() > 0:
            return output_counts.argmax().item()
        return random.randint(0, self.n_output - 1)

    def update(
        self, action: int, reward: float, next_obs: np.ndarray, done: bool,
        grid_size: int
    ):
        nov = self.novelty_module(tuple(int(x) for x in next_obs))
        modulated_reward = reward + self.novelty_coeff * nov
        delta = modulated_reward - self.baseline
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * modulated_reward
        self.policy.update_weights(delta)

        if done:
            self.policy.reset_traces()
            self.novelty_module.reset()

    def reset(self):
        self.policy.reset_traces()
        self.policy.reset_state()
        self.novelty_module.reset()
        self.baseline = 0.0

    def set_weights(self, weights):
        for i, layer in enumerate(self.policy.layers):
            layer.weight.data = weights[i]


def train_episode(env: GridWorld, agent: SpikeAgent, max_steps: int = 200) -> dict:
    """Run a single training episode and return metrics."""
    obs = env.reset()
    agent.reset()
    total_reward = 0.0
    path = [env.agent_pos]
    rewards = []
    spike_counts = []
    success = False

    for step in range(max_steps):
        action = agent.act(obs, env.size)
        next_obs, reward, done, info = env.step(action)
        agent.update(action, reward, next_obs, done, env.size)
        total_reward += reward
        path.append(env.agent_pos)
        rewards.append(reward)
        obs = next_obs

        if done:
            if env.agent_pos == env.goal_pos:
                success = True
            break

    return {
        "success": success,
        "total_reward": total_reward,
        "steps": len(rewards),
        "path": path,
    }