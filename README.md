# SpikeQuest: Path to AGI with Spiking Neural Networks

## Overview
SpikeQuest is a research project exploring artificial general intelligence (AGI) using spiking neural networks (SNNs). It simulates an intelligent agent navigating a 5x5 grid world from (0,0) to (4,4), aiming to mimic AGI-like behavior through novelty, reward-based learning, and pathfinding. The agent uses spike-timing-dependent plasticity (STDP), rewards, and exploration to learn optimal paths while avoiding obstacles.

## Files
- `spikequestAGI.py`: Core SNN implementation for grid navigation, reaching the goal at step 19.
- `netxspikequestAGI.py`: Enhanced version with grid visualization and advanced features, showing the agent’s path progress (e.g., animated steps on a 5x5 grid).
- `spikequestAGI-THINK.py`: Experimental script for deeper AGI concepts, including step-like progress tracking and theoretical exploration of SNN dynamics.

## Getting Started
1. Clone this repository:
   git clone https://github.com/NullLabTests/SpikeQuest.git
2. Install dependencies:
   pip install numpy matplotlib networkx
3. Run:
- For core navigation: `python spikequestAGI.py`
- For visualizations: `python netxspikequestAGI.py`

## Visualizations
- **Step-Like Graph**: Tracks cumulative progress, reaching a target value (e.g., 4.0) at step 19, as shown in Figure 1.
- **Grid Visualization**: Displays the agent’s path on a 5x5 grid, color-coded (green for new states, orange for visited, red for obstacles, green for the goal).

### Screenshots
#### SpikeQuest THINK Visualization
![SpikeQuest THINK Visualization](https://i.imgur.com/ShQv8Z6.png)
This image shows the output of `spikequestAGI-THINK.py`, illustrating the step-like progress of the SNN toward the goal at step 19. The graph tracks cumulative value over scaled steps, providing insight into the agent’s learning and exploration behavior in pursuit of AGI-like capabilities.

#### SpikeQuest Net Animation
![SpikeQuest Net Animation](https://i.imgur.com/AhocsfV.png)
This image captures the animated grid visualization from `netxspikequestAGI.py` at Step 14, showing the agent’s path through a 5x5 grid. The color-coded grid highlights new states (green), visited states (orange), obstacles (red), and the goal (green), offering a dynamic view of the SNN’s navigation progress.

## License
This project is licensed under the MIT License - see the [LICENSE](#license) file for details.
