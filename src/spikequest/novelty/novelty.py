import torch
import numpy as np


class VisitedNovelty:
    """Visited-state novelty bonus.

    Returns 1.0 the first time a state is seen, 0.0 thereafter.
    Simple but often effective for exploration.
    """

    def __init__(self):
        self.visited = set()

    def reset(self):
        self.visited.clear()

    def __call__(self, state: tuple) -> float:
        if state not in self.visited:
            self.visited.add(state)
            return 1.0
        return 0.0


class PredictionErrorNovelty:
    """Prediction-error novelty module.

    A small forward-model network learns to predict the next
    observation from current (obs, action).  The prediction error
    (MSE) serves as a novelty bonus: states the model cannot
    predict well are deemed novel.

    This is a simplified form of the "intrinsic curiosity module"
    (Pathak et al., 2017).

    Args:
        obs_dim: dimension of observation features
        n_actions: number of discrete actions
        hidden_dim: hidden layer size
        lr: learning rate for the predictor
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 4,
        hidden_dim: int = 32,
        lr: float = 0.001,
    ):
        encoding_dim = obs_dim + n_actions
        self.net = torch.nn.Sequential(
            torch.nn.Linear(encoding_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, obs_dim),
        )
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def reset(self):
        pass

    def __call__(
        self, obs: torch.Tensor, action: int, next_obs: torch.Tensor
    ) -> float:
        action_onehot = torch.zeros(4)
        action_onehot[action] = 1.0
        inp = torch.cat([obs, action_onehot])
        pred = self.net(inp)
        error = torch.nn.functional.mse_loss(pred, next_obs, reduction="sum")

        self.optimizer.zero_grad()
        error.backward()
        self.optimizer.step()

        return error.item()


class DualTimescaleNovelty:
    """Dual-timescale novelty with fast excitation and slow suppression.

    Inspired by neuromodulatory systems (acetylcholine / serotonin)
    that implement explore-exploit trade-offs.

    The fast trace reacts immediately to raw novelty.
    The slow trace tracks a running baseline.
    Novelty = max(0, fast - slow).

    When novelty is consistently above baseline, fast > slow -> explore.
    When novelty saturates, fast ~= slow -> suppress novelty -> exploit.

    Args:
        tau_fast: time constant for fast trace (timesteps)
        tau_slow: time constant for slow trace (timesteps)
        dt: simulation timestep
    """

    def __init__(self, tau_fast: float = 10.0, tau_slow: float = 200.0, dt: float = 1.0):
        self.alpha_fast = np.exp(-dt / tau_fast)
        self.alpha_slow = np.exp(-dt / tau_slow)
        self.fast = 0.0
        self.slow = 0.0

    def reset(self):
        self.fast = 0.0
        self.slow = 0.0

    def __call__(self, raw_novelty: float) -> float:
        self.fast = self.alpha_fast * self.fast + (1 - self.alpha_fast) * raw_novelty
        self.slow = self.alpha_slow * self.slow + (1 - self.alpha_slow) * raw_novelty
        return max(0.0, self.fast - self.slow)