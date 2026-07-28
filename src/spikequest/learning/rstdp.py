import torch
import numpy as np


class RSTDP:
    """Reward-modulated STDP with eligibility traces (three-factor rule).

    Implements the three-factor learning rule combining:
      1. Pre-synaptic trace (low-pass filtered spike train)
      2. Post-synaptic trace
      3. Eligibility trace that accumulates STDP events
      4. Global neuromodulator that converts eligibility into weight change

    --- Traces (updated every simulation timestep) ---

        pre_trace[t+1]  = alpha_pre * pre_trace[t]  + S_pre[t]

        post_trace[t+1] = alpha_post * post_trace[t] + S_post[t]

        where alpha = exp(-dt / tau) and S is the spike vector {0,1}.

    --- Eligibility trace (per synapse) ---

        e[t+1] = alpha_e * e[t]
                + pre_trace[t] * S_post[t]        (LTP eligibility)
                - post_trace[t] * S_pre[t]         (LTD eligibility)

        The first term says: when post-synaptic neuron j fires,
        weight w_ij becomes eligible for potentiation proportional to
        recent pre-synaptic activity.

        The second term says: when pre-synaptic neuron i fires,
        weight w_ij becomes eligible for depression proportional to
        recent post-synaptic activity.

        This is the nearest-neighbour STDP pairing scheme captured
        in eligibility form (Izhikevich, 2007; Fremaux & Gerstner, 2016).

    --- Weight update (when modulator M(t) arrives) ---

        Delta w = lr * M(t) * e(t)

        where M(t) is the neuromodulatory signal (reward, TD-error, etc.).

    --- Homeostasis ---

        Optional weight decay: Delta_w -= lr * wd * w

    References:
        Izhikevich, E. M. (2007). Solving the distal reward problem
            through linkage of STDP and dopamine signaling. Cerebral Cortex.
        Fremaux, N. & Gerstner, W. (2016). Reward-modulated STDP.
            Frontiers in Synaptic Neuroscience.
        Rombouts, J. et al. (2015). A learning rule for place-field
            formation from STDP and reward-modulated plasticity.
            PLOS Computational Biology.
    """

    def __init__(
        self,
        tau_pre: float = 20.0,
        tau_post: float = 20.0,
        tau_elig: float = 50.0,
        lr: float = 0.005,
        lr_decay: float = 1.0,
        w_init: float = 0.3,
        w_init_std: float = 0.1,
        w_min: float = 0.0,
        w_max: float = 1.0,
        weight_decay: float = 0.0,
        dt: float = 1.0,
        device: str = "cpu",
    ):
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.tau_elig = tau_elig
        self.lr = lr
        self.lr_decay = lr_decay
        self.w_init = w_init
        self.w_init_std = w_init_std
        self.w_min = w_min
        self.w_max = w_max
        self.weight_decay = weight_decay
        self.dt = dt
        self.device = device

        self.alpha_pre = np.exp(-dt / tau_pre)
        self.alpha_post = np.exp(-dt / tau_post)
        self.alpha_elig = np.exp(-dt / tau_elig)

    def init_weight(self, n_pre: int, n_post: int) -> torch.Tensor:
        w = self.w_init + self.w_init_std * torch.randn(n_pre, n_post, device=self.device)
        return w.clamp(self.w_min, self.w_max)

    @staticmethod
    def update_trace(trace: torch.Tensor, alpha: float, spikes: torch.Tensor) -> torch.Tensor:
        return alpha * trace + spikes

    def eligibility(
        self,
        elig: torch.Tensor,
        pre_trace: torch.Tensor,
        post_trace: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
    ) -> torch.Tensor:
        ltp = pre_trace.unsqueeze(-1) * post_spikes.unsqueeze(-2)
        ltd = post_trace.unsqueeze(-2) * pre_spikes.unsqueeze(-1)
        stdp_signal = ltp - ltd
        return self.alpha_elig * elig + stdp_signal

    def update_weight(
        self,
        weight: torch.Tensor,
        elig: torch.Tensor,
        modulator: float,
    ) -> torch.Tensor:
        delta = self.lr * modulator * elig
        if self.weight_decay > 0.0:
            delta -= self.lr * self.weight_decay * weight
        weight = weight + delta
        return weight.clamp(self.w_min, self.w_max)

    def decay_lr(self):
        self.lr *= self.lr_decay

    def reset_traces(self, n_pre: int, n_post: int):
        return {
            "pre_trace": torch.zeros(n_pre, device=self.device),
            "post_trace": torch.zeros(n_post, device=self.device),
            "elig": torch.zeros(n_pre, n_post, device=self.device),
        }


class TDRSTDP(RSTDP):
    """TD-modulated STDP with learned state-value baseline.

    Uses TD-error delta = R + gamma * V(s') - V(s) as the
    neuromodulator instead of the raw reward.

    A small value network (linear or shallow MLP) is trained
    alongside the policy using TD learning.

    Args:
        n_state: dimension of state features for the value network
        gamma: discount factor
        lr_critic: learning rate for the value network
    """

    def __init__(
        self,
        n_state: int,
        gamma: float = 0.95,
        lr_critic: float = 0.001,
        hidden_critic: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(n_state, hidden_critic),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_critic, 1),
        )
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    def td_error(self, state: torch.Tensor, reward: float, next_state: torch.Tensor, done: bool) -> float:
        V_s = self.critic(state.unsqueeze(0)).squeeze()
        with torch.no_grad():
            V_next = self.critic(next_state.unsqueeze(0)).squeeze() if not done else 0.0
        delta = reward + self.gamma * V_next - V_s

        loss = delta ** 2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return delta.item()