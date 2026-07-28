import numpy as np
from typing import List, Tuple


def compute_metrics(
    successes: List[bool],
    steps_list: List[int],
    rewards_list: List[float],
    path_lengths: List[int],
    spike_rates: List[float] = None,
) -> dict:
    """Compute summary metrics across multiple evaluation episodes.

    Args:
        successes: boolean success per episode
        steps_list: steps to termination per episode
        rewards_list: cumulative reward per episode
        path_lengths: number of unique states visited
        spike_rates: average spike rate per episode (optional)

    Returns:
        dict of metric names to (mean, std) tuples
    """
    results = {}
    results["success_rate"] = (float(np.mean(successes)), float(np.std(successes)))
    results["steps_to_goal"] = (
        float(np.mean([s for s, succ in zip(steps_list, successes) if succ])),
        float(np.std([s for s, succ in zip(steps_list, successes) if succ]))
    ) if any(successes) else (float("inf"), float("inf"))
    results["cumulative_reward"] = (float(np.mean(rewards_list)), float(np.std(rewards_list)))
    results["path_length"] = (float(np.mean(path_lengths)), float(np.std(path_lengths)))
    if spike_rates is not None:
        results["spike_rate"] = (float(np.mean(spike_rates)), float(np.std(spike_rates)))
    return results


def sparseness_score(spike_trains: np.ndarray) -> float:
    """Compute population sparseness.

    Defined as (1 - (sum(r_i/N))^2 / sum(r_i^2/N)) where r_i
    is the firing rate of neuron i.  Ranges from 0 (dense) to 1 (sparse).
    """
    r = spike_trains.mean(axis=0)
    n = len(r)
    numerator = (r.sum() / n) ** 2
    denominator = (r ** 2).sum() / n
    return 1.0 - numerator / denominator if denominator > 0 else 0.0


def estimate_energy(synops: int, n_neurons: int) -> float:
    """Rough energy proxy: number of synaptic operations (synops).

    Each spike contributes O(n_pre) synaptic operations.
    For a single-layer network, total synops = n_spikes * n_pre.
    """
    return float(synops)