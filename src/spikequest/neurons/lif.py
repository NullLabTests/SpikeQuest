import torch
import numpy as np
from typing import Optional


class LIFLayer(torch.nn.Module):
    """Leaky Integrate-and-Fire neuron layer.

    Continuous-time dynamics:
        tau_m * dV/dt = -V + I_syn

    Discrete-time update (exact exponential integration):
        V[t+1] = beta * V[t] + (1 - beta) * I_syn[t]
        where beta = exp(-dt / tau_m)

    If V >= V_th: emit spike, V <- V_reset.

    Args:
        n_in: number of input features
        n_out: number of output neurons
        tau_m: membrane time constant (ms)
        V_th: firing threshold
        V_reset: reset potential
        dt: simulation timestep (ms)
        weight_init: weight initialisation scale
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        tau_m: float = 20.0,
        V_th: float = 1.0,
        V_reset: float = 0.0,
        dt: float = 1.0,
        weight_init: float = 0.3,
        device: str = "cpu",
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.V_th = V_th
        self.V_reset = V_reset
        self.dt = dt
        self.beta = np.exp(-dt / tau_m)

        self.weight = torch.nn.Parameter(
            torch.empty(n_in, n_out, device=device)
        )
        self.bias = torch.nn.Parameter(torch.zeros(n_out, device=device))
        torch.nn.init.uniform_(self.weight, -weight_init, weight_init)

        self.register_buffer("V", torch.zeros(n_out, device=device))
        self.register_buffer("spikes", torch.zeros(n_out, device=device))

    def reset_state(self):
        self.V.zero_()
        self.spikes.zero_()

    def forward(self, I_syn: torch.Tensor) -> torch.Tensor:
        current = I_syn.squeeze() @ self.weight + self.bias
        self.V = self.beta * self.V + (1 - self.beta) * current
        self.spikes = (self.V >= self.V_th).float()
        self.V = torch.where(self.spikes > 0,
                             torch.full_like(self.V, self.V_reset), self.V)
        return self.spikes


class ALIFLayer(torch.nn.Module):
    """Adaptive LIF neuron layer with spike-frequency adaptation.

    Adds an adaptation current w that increases after each spike
    and decays with time constant tau_w, effectively raising the
    effective threshold after high-frequency firing.

    Dynamics:
        tau_m * dV/dt = -V + I_syn - w
        tau_w * dw/dt = -w + b * S(t)

        where S(t) is the spike train and b is the spike-triggered
        adaptation increment.

    Args:
        n_in: input dimension
        n_out: output dimension
        tau_m: membrane time constant
        tau_w: adaptation time constant
        a: adaptation coupling strength (subthreshold)
        b: spike-triggered adaptation increment
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        tau_m: float = 20.0,
        tau_w: float = 300.0,
        a: float = 0.0,
        b: float = 0.05,
        V_th: float = 1.0,
        V_reset: float = 0.0,
        dt: float = 1.0,
        weight_init: float = 0.3,
        device: str = "cpu",
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.V_th = V_th
        self.V_reset = V_reset
        self.dt = dt
        self.a = a
        self.b = b
        self.beta = np.exp(-dt / tau_m)
        self.beta_w = np.exp(-dt / tau_w)

        self.weight = torch.nn.Parameter(
            torch.empty(n_in, n_out, device=device)
        )
        self.bias = torch.nn.Parameter(torch.zeros(n_out, device=device))
        torch.nn.init.uniform_(self.weight, -weight_init, weight_init)

        self.register_buffer("V", torch.zeros(n_out, device=device))
        self.register_buffer("w", torch.zeros(n_out, device=device))
        self.register_buffer("spikes", torch.zeros(n_out, device=device))

    def reset_state(self):
        self.V.zero_()
        self.w.zero_()
        self.spikes.zero_()

    def forward(self, I_syn: torch.Tensor) -> torch.Tensor:
        current = I_syn.squeeze() @ self.weight + self.bias
        I_eff = current - self.a * self.w
        self.V = self.beta * self.V + (1 - self.beta) * I_eff
        self.spikes = (self.V >= self.V_th).float()
        self.w = self.beta_w * self.w + self.b * self.spikes
        self.V = torch.where(self.spikes > 0,
                             torch.full_like(self.V, self.V_reset), self.V)
        return self.spikes