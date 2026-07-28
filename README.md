# SpikeQuest

**Spiking Neural Network agent for grid-world navigation using reward-modulated STDP and novelty-driven exploration.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## Scientific Motivation

Biological learning operates under severe constraints: neurons communicate through discrete spikes, synaptic plasticity depends only on locally available information (pre- and post-synaptic activity), and reward signals are sparse and delayed. Despite these constraints, animals learn complex navigation behaviours efficiently.

SpikeQuest implements a minimal, biologically plausible model of this process:

- **Spiking neurons** (LIF) communicate through all-or-nothing events, respecting the biological constraint of discrete communication.
- **Three-factor plasticity** (reward-modulated STDP) uses only local spike timing and a global neuromodulator, mirroring the role of dopamine in reinforcement learning.
- **Novelty-driven exploration** provides an intrinsic motivation signal, encouraging the agent to visit under-explored states.

The goal is **not** to beat deep RL baselines on complex tasks, but to provide a clean, reproducible framework for studying how local plasticity rules can solve reinforcement learning problems under biologically realistic constraints.

## Methods

### Neuron Model: Leaky Integrate-and-Fire (LIF)

$$
\tau_m \frac{dV}{dt} = -V + I_{\text{syn}}(t); \quad \text{if } V \geq V_{\text{th}}: \text{spike}, V \leftarrow V_{\text{reset}}
$$

Implemented as exact exponential integration:
$$V[t+1] = \beta V[t] + (1-\beta) I_{\text{syn}}[t], \quad \beta = \exp(-\Delta t / \tau_m)$$

An adaptive variant (ALIF) adds spike-frequency adaptation via a slow after-hyperpolarisation current.

### Plasticity Rule: Three-Factor R-STDP with Eligibility Traces

The core learning rule combines three factors:

**Factor 1 — Pre-synaptic trace:**
$$\bar{x}_i[t+1] = \alpha_{\text{pre}} \bar{x}_i[t] + S_{\text{pre},i}[t], \quad \alpha_{\text{pre}} = \exp(-\Delta t / \tau_{\text{pre}})$$

**Factor 2 — Post-synaptic trace:**
$$\bar{y}_j[t+1] = \alpha_{\text{post}} \bar{y}_j[t] + S_{\text{post},j}[t], \quad \alpha_{\text{post}} = \exp(-\Delta t / \tau_{\text{post}})$$

**Factor 3 — Eligibility trace (accumulates STDP events):**
$$e_{ij}[t+1] = \alpha_e e_{ij}[t] + \bar{x}_i[t] S_{\text{post},j}[t] - \bar{y}_j[t] S_{\text{pre},i}[t]$$

The first term ($\bar{x}_i S_{\text{post},j}$) marks the synapse for potentiation when post-synaptic firing follows pre-synaptic activity (LTP). The second term ($\bar{y}_j S_{\text{pre},i}$) marks it for depression when pre-synaptic firing follows post-synaptic activity (LTD).

**Weight update (neuromodulator-gated):**
$$\Delta w_{ij} = \eta \cdot M(t) \cdot e_{ij}(t)$$

where $M(t)$ is the neuromodulatory signal:
- **Simple:** $M = R + \eta_{\text{nov}} \cdot \text{novelty} - \text{baseline}$
- **TD-modulated:** $M = \delta = R + \gamma V(s') - V(s)$

The eligibility trace acts as a *synaptic memory*: it accumulates STDP-relevant activity and only converts to a weight change when a delayed reward signal arrives, addressing the distal reward problem (Izhikevich, 2007).

### Novelty and Exploration

| Module | Mechanism | Reference |
|--------|-----------|-----------|
| VisitedSet | Binary: 1 if unseen, 0 if visited | Classic RL |
| PredictionError | MSE of learned forward model | Pathak et al. (2017) |
| DualTimescale | Fast excitation - slow suppression | Neuromodulatory inspiration |

All novelty signals can be combined with the task reward to form the neuromodulator $M(t)$.

### Encoding

Position $(x, y)$ is population-coded as a spike-rate vector of length $2N$ (one-hot in each coordinate, repeated as Poisson spike trains over $T$ simulation timesteps). For partially observed settings, a flattened local view patch replaces the position encoding.

## Repository Structure

```
spikequest/
├── pyproject.toml              # Build config & dependencies
├── README.md
├── LICENSE
├── configs/
│   └── experiment.yaml         # Default experiment config
├── src/
│   └── spikequest/
│       ├── env/
│       │   └── grid_world.py   # Gymnasium-style GridWorld
│       ├── neurons/
│       │   ├── lif.py          # LIF and ALIF layers
│       ├── learning/
│       │   └── rstdp.py        # R-STDP & TD-RSTDP rules
│       ├── novelty/
│       │   └── novelty.py      # Novelty modules
│       ├── networks/
│       │   └── snn_policy.py   # Feedforward SNN policy
│       ├── agents/
│       │   └── spike_agent.py  # Full agent + training loop
│       └── utils/
│           ├── metrics.py      # Evaluation metrics
│           ├── logging.py      # CSV logging
│           └── seeding.py      # Reproducible seeding
├── experiments/
│   ├── run_experiment.py       # Multi-seed experiment runner
│   ├── run_comparison.py       # SNN vs tabular Q comparison
│   └── demo.py                 # Walkthrough script
├── tests/
│   ├── test_neurons.py
│   ├── test_rstdp.py
│   └── test_env.py
└── notebooks/
    └── spikequest_demo.ipynb   # Interactive walkthrough
```

## Quick Start

```bash
# Install
pip install -e .

# Run a multi-seed experiment with comparison to tabular Q-learning
python experiments/run_comparison.py --seeds 10 --episodes 300

# Run the walkthrough demo
python experiments/demo.py
```

## Reproducing Experiments

```bash
# Default experiment (5 seeds, 200 episodes)
python experiments/run_experiment.py --config configs/experiment.yaml --seeds 5

# Comparison with tabular Q-learning (10 seeds, 300 episodes)
python experiments/run_comparison.py --seeds 10 --episodes 300
```

Results (learning curves, final metrics) are saved to `experiments/outputs/`.

## Extending

- **New environments:** subclass or modify `GridWorld` in `env/grid_world.py`; the agent expects a Gymnasium-like API.
- **New learning rules:** add to `learning/` following the `RSTDP` interface (update traces, compute eligibility, update weights).
- **New novelty modules:** add to `novelty/` with a callable interface returning a scalar bonus.
- **snnTorch backend:** install with `pip install -e ".[snntorch]"` and swap neuron layers for snnTorch's `snn.Leaky` with surrogate gradients.

## Citation

If you use this code in a research publication, please cite:

```bibtex
@software{spikequest2026,
  author = {SpikeQuest Contributors},
  title = {SpikeQuest: SNN Grid-World Navigation with R-STDP},
  year = {2026},
  url = {https://github.com/NullLabTests/SpikeQuest}
}
```

## Limitations (Honest Assessment)

1. **R-STDP converges slower** than backprop-based RL on dense-reward tasks due to the locality constraint.
2. **Grid-world is a toy domain** — scaling to continuous control or vision-based tasks requires additional machinery (event-based sensors, larger networks, more sophisticated encoding).
3. **The visited-set novelty** is a heuristic; prediction-error novelty is more theoretically grounded but computationally more expensive.
4. **No hardware deployment** — the current implementation is pure PyTorch and has not been tested on neuromorphic hardware (Loihi, SpiNNaker).
5. **Single-agent, single-goal** — the framework does not yet address multi-agent, multi-goal, or continual learning scenarios.

## Roadmap

- [ ] Continuous control environments (PENDULUM, HalfCheetah via event-based simulation)
- [ ] Loihi / Lava backend for neuromorphic deployment
- [ ] Hierarchical SNN with temporal replay
- [ ] Surrogate-gradient training as an alternative to R-STDP
- [ ] Cognitive map formation via grid-cell / place-cell encoding

## License

MIT — see `LICENSE`.