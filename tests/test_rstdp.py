import torch
import numpy as np
from spikequest.learning.rstdp import RSTDP


class TestRSTDP:
    def setup_method(self):
        self.rstdp = RSTDP(tau_pre=20.0, tau_post=20.0, tau_elig=50.0,
                           lr=0.01, w_init=0.5, w_init_std=0.0)

    def test_init_weight(self):
        w = self.rstdp.init_weight(3, 4)
        assert w.shape == (3, 4)
        assert torch.allclose(w, torch.tensor(0.5))

    def test_trace_update(self):
        trace = torch.zeros(5)
        spikes = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])
        new_trace = RSTDP.update_trace(trace, self.rstdp.alpha_pre, spikes)
        assert new_trace.shape == (5,)
        assert new_trace[0] == 1.0
        assert new_trace[2] == 1.0
        assert new_trace[1] == 0.0

    def test_trace_decay(self):
        trace = torch.ones(3)
        spikes = torch.zeros(3)
        new_trace = RSTDP.update_trace(trace, 0.5, spikes)
        assert torch.allclose(new_trace, torch.tensor([0.5, 0.5, 0.5]))

    def test_eligibility(self):
        w = self.rstdp.init_weight(2, 2)
        pre_trace = torch.tensor([1.0, 0.5])
        post_trace = torch.tensor([0.3, 0.8])
        pre_spikes = torch.tensor([0.0, 1.0])
        post_spikes = torch.tensor([1.0, 0.0])
        elig = torch.zeros(2, 2)

        new_elig = self.rstdp.eligibility(
            elig, pre_trace, post_trace, pre_spikes, post_spikes
        )

        assert new_elig.shape == (2, 2)
        # LTP: pre_trace * post_spikes
        # post_spikes = [1, 0], pre_trace = [1.0, 0.5]
        # ltp[0,:] = [1.0*1, 1.0*0] = [1, 0]
        # ltp[1,:] = [0.5*1, 0.5*0] = [0.5, 0]
        # LTD: post_trace * pre_spikes
        # pre_spikes = [0, 1], post_trace = [0.3, 0.8]
        # ltd[:,0] = [0.3*0, 0.3*1] = [0, 0.3]
        # ltd[:,1] = [0.8*0, 0.8*1] = [0, 0.8]
        expected = torch.tensor([[1.0, 0.0], [0.5 - 0.3, 0.0 - 0.8]])
        assert torch.allclose(new_elig, expected, atol=1e-6)

    def test_weight_update(self):
        w = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        elig = torch.tensor([[1.0, -0.5], [0.0, 2.0]])
        modulator = 1.0
        new_w = self.rstdp.update_weight(w, elig, modulator)
        expected = w + 0.01 * elig
        assert torch.allclose(new_w, expected)

    def test_negative_modulator(self):
        w = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        elig = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        new_w = self.rstdp.update_weight(w, elig, -1.0)
        expected = w - 0.01 * elig
        assert torch.allclose(new_w, expected)

    def test_weight_clamp(self):
        rstdp = RSTDP(w_min=0.0, w_max=1.0, lr=100.0)
        w = torch.tensor([[0.5, 0.5]])
        elig = torch.tensor([[1.0, -1.0]])
        new_w = rstdp.update_weight(w, elig, 1.0)
        assert new_w.min() >= 0.0
        assert new_w.max() <= 1.0

    def test_reset_traces(self):
        traces = self.rstdp.reset_traces(3, 4)
        assert "pre_trace" in traces
        assert "post_trace" in traces
        assert "elig" in traces
        assert traces["pre_trace"].shape == (3,)
        assert traces["post_trace"].shape == (4,)
        assert traces["elig"].shape == (3, 4)
        assert torch.all(traces["pre_trace"] == 0.0)
        assert torch.all(traces["post_trace"] == 0.0)
        assert torch.all(traces["elig"] == 0.0)

    def test_weight_decay(self):
        rstdp = RSTDP(lr=0.01, weight_decay=0.1)
        w = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        elig = torch.zeros(2, 2)
        new_w = rstdp.update_weight(w, elig, 0.0)
        # weight_decay should reduce weights
        assert torch.all(new_w < w)