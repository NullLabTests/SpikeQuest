import torch
import pytest
from spikequest.neurons.lif import LIFLayer, ALIFLayer


class TestLIFLayer:
    def test_init(self):
        layer = LIFLayer(10, 20)
        assert layer.weight.shape == (10, 20)
        assert layer.bias.shape == (20,)
        assert layer.V.shape == (20,)

    def test_forward_output_shape(self):
        layer = LIFLayer(5, 3)
        x = torch.randn(5)
        out = layer(x)
        assert out.shape == (3,)
        assert out.dtype == torch.float32

    def test_spike_or_no_spike(self):
        layer = LIFLayer(2, 1, V_th=0.5, weight_init=1.0)
        layer.weight.data = torch.tensor([[10.0], [10.0]])
        torch.nn.init.constant_(layer.bias, 0.0)
        # subthreshold input
        out = layer(torch.zeros(2))
        assert out.item() == 0.0
        # suprathreshold input (10+10=20, (1-beta)*20 ≈ 0.98)
        layer.reset_state()
        layer.V_th = 0.5
        out = layer(torch.tensor([1.0, 1.0]))
        assert out.item() == 1.0

    def test_reset_state(self):
        layer = LIFLayer(5, 5)
        layer(torch.randn(5))
        layer.reset_state()
        assert torch.all(layer.V == 0.0)
        assert torch.all(layer.spikes == 0.0)

    def test_persistence(self):
        layer = LIFLayer(2, 1, tau_m=10.0, V_th=5.0)
        layer.weight.data = torch.tensor([[1.0], [0.0]])
        torch.nn.init.constant_(layer.bias, 0.0)
        x = torch.tensor([2.0, 0.0])
        out1 = layer(x)
        v1 = layer.V.clone()
        out2 = layer(x)
        assert torch.all(layer.V >= v1 - 1e-6)


class TestALIFLayer:
    def test_init(self):
        layer = ALIFLayer(5, 10, a=0.1, b=0.05)
        assert layer.weight.shape == (5, 10)
        assert layer.w.shape == (10,)

    def test_adaptation(self):
        layer = ALIFLayer(2, 1, a=0.2, b=0.1, tau_m=5.0, tau_w=50.0, V_th=1.0)
        layer.weight.data = torch.tensor([[5.0], [5.0]])
        strong = torch.tensor([1.0, 1.0])
        # fire several times
        for _ in range(10):
            layer(strong)
        # adaptation current should have built up
        assert layer.w.item() > 0.0

    def test_reset_state(self):
        layer = ALIFLayer(5, 5, a=0.1, b=0.05)
        layer(torch.randn(5))
        layer.reset_state()
        assert torch.all(layer.V == 0.0)
        assert torch.all(layer.w == 0.0)
        assert torch.all(layer.spikes == 0.0)