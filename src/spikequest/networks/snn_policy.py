import torch
import numpy as np
from typing import List, Optional

from ..neurons.lif import LIFLayer
from ..learning.rstdp import RSTDP


class SNNPolicy(torch.nn.Module):
    """Feedforward SNN policy network with R-STDP learning.

    Architecture:
        Input (rate-coded) -> LIF hidden layer(s) -> LIF output layer

    The observation is encoded as a population-coded spike pattern
    that persists for T simulation timesteps per environment step.
    The output neuron with the highest spike count determines the action.

    Weight matrices are stored as nn.Parameters but updated by the
    external R-STDP rule (not by gradient descent).

    Args:
        n_input: number of input features
        n_hidden: hidden layer size(s); int or list of ints
        n_output: number of output neurons (= action dim)
        T: number of simulation timesteps per environment step
        neuron_config: dict passed to LIFLayer constructor
        rstdp_config: dict passed to RSTDP constructor
    """

    def __init__(
        self,
        n_input: int = 20,
        n_hidden: int = 64,
        n_output: int = 4,
        T: int = 10,
        neuron_config: Optional[dict] = None,
        rstdp_config: Optional[dict] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.T = T
        self.device = device

        nc = {} if neuron_config is None else neuron_config
        rc = {} if rstdp_config is None else rstdp_config

        if isinstance(n_hidden, int):
            n_hidden = [n_hidden]
        self.layer_sizes = [n_input] + list(n_hidden) + [n_output]

        self.layers = torch.nn.ModuleList()
        for i in range(len(self.layer_sizes) - 1):
            self.layers.append(
                LIFLayer(self.layer_sizes[i], self.layer_sizes[i + 1], **nc)
            )

        self.rstdp = RSTDP(**rc)
        self.traces = {}
        self._init_traces()

    def _init_traces(self):
        self.traces = {}
        for i in range(len(self.layer_sizes) - 1):
            key = f"l{i}"
            self.traces[key] = self.rstdp.reset_traces(
                self.layer_sizes[i], self.layer_sizes[i + 1]
            )

    def reset_state(self):
        for layer in self.layers:
            layer.reset_state()

    def reset_traces(self):
        self._init_traces()

    def forward_sim(
        self, encoded_obs: torch.Tensor
    ) -> torch.Tensor:
        """Run forward pass for all T timesteps, return cumulative output spikes.

        Also updates eligibility traces internally.
        """
        output_counts = torch.zeros(self.n_output, device=self.device)

        for _ in range(self.T):
            x = encoded_obs
            layer_spikes = [x]

            for i, layer in enumerate(self.layers):
                x = layer(x)
                layer_spikes.append(x)

            output_counts += layer_spikes[-1]

            for i in range(len(self.layers)):
                pre_spikes = layer_spikes[i]
                post_spikes = layer_spikes[i + 1]
                t = self.traces[f"l{i}"]

                t["pre_trace"] = self.rstdp.update_trace(
                    t["pre_trace"], self.rstdp.alpha_pre, pre_spikes
                )
                t["post_trace"] = self.rstdp.update_trace(
                    t["post_trace"], self.rstdp.alpha_post, post_spikes
                )
                t["elig"] = self.rstdp.eligibility(
                    t["elig"],
                    t["pre_trace"],
                    t["post_trace"],
                    pre_spikes,
                    post_spikes,
                )

        return output_counts

    def get_weights(self) -> List[torch.Tensor]:
        return [layer.weight.data for layer in self.layers]

    def update_weights(self, modulator: float):
        for i, layer in enumerate(self.layers):
            t = self.traces[f"l{i}"]
            layer.weight.data = self.rstdp.update_weight(
                layer.weight.data, t["elig"], modulator
            )

    def compute_encoding(self, obs: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(obs).float().to(self.device)