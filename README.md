# SpikeQuest: Path to AGI with Spiking Neural Networks

## Overview
SpikeQuest is a research project exploring artificial general intelligence (AGI) using spiking neural networks (SNNs). It simulates an intelligent agent navigating a 5x5 grid world from (0,0) to (4,4), aiming to mimic AGI-like behavior through novelty, reward-based learning, and pathfinding. The agent uses STDP, rewards, and exploration to learn optimal paths while avoiding obstacles.

## Files
- : Core SNN implementation for grid navigation, reaching the goal at step 19.
- : Enhanced version with grid visualization and advanced features, showing path progress (e.g., Step 14 grid).
- : Experimental script for deeper AGI concepts, including step-like progress tracking.

## Getting Started
1. Clone this repository: 
2. Install dependencies: Requirement already satisfied: numpy in /home/illy/miniconda3/lib/python3.11/site-packages (1.26.4)
Requirement already satisfied: matplotlib in /home/illy/miniconda3/lib/python3.11/site-packages (3.9.4)
Requirement already satisfied: networkx in /home/illy/miniconda3/lib/python3.11/site-packages (3.4.2)
Requirement already satisfied: contourpy>=1.0.1 in /home/illy/miniconda3/lib/python3.11/site-packages (from matplotlib) (1.2.0)
Requirement already satisfied: cycler>=0.10 in /home/illy/miniconda3/lib/python3.11/site-packages (from matplotlib) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /home/illy/miniconda3/lib/python3.11/site-packages (from matplotlib) (4.49.0)
Requirement already satisfied: kiwisolver>=1.3.1 in /home/illy/miniconda3/lib/python3.11/site-packages (from matplotlib) (1.4.5)
Requirement already satisfied: packaging>=20.0 in /home/illy/miniconda3/lib/python3.11/site-packages (from matplotlib) (24.0)
Requirement already satisfied: pillow>=8 in /home/illy/miniconda3/lib/python3.11/site-packages (from matplotlib) (10.4.0)
Requirement already satisfied: pyparsing>=2.3.1 in /home/illy/miniconda3/lib/python3.11/site-packages (from matplotlib) (3.1.2)
Requirement already satisfied: python-dateutil>=2.7 in /home/illy/miniconda3/lib/python3.11/site-packages (from matplotlib) (2.9.0.post0)
Requirement already satisfied: six>=1.5 in /home/illy/miniconda3/lib/python3.11/site-packages (from python-dateutil>=2.7->matplotlib) (1.16.0)
3. Run: Goal reached at step 19! or Goal reached at step 33!
Path: [(0, 0), (1, 0), (2, 0), (3, 0), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 4), (3, 4), (3, 3), (3, 2), (3, 3), (2, 3), (1, 3), (2, 3), (2, 3), (3, 3), (2, 3), (2, 3), (3, 3), (2, 3), (2, 3), (2, 4), (3, 4), (2, 4), (2, 4), (2, 3), (2, 4), (3, 4), (2, 4), (3, 4), (4, 4)]
Total Reward: 3.0 for visualizations.

## Visualizations
- **Step-Like Graph**: Tracks cumulative progress (e.g., Figure 1, reaching 4.0 at step 19).
- **Grid Visualization**: Shows the agent’s path on a 5x5 grid, color-coded (green=new, orange=visited, red=obstacle, green=goal).

## License
This project is licensed under the MIT License - see the [LICENSE](#license) file for details.

## Contributors
- Illy (lead developer)

## Status
- Goal reached at step 19 in multiple runs.
- Current challenge: Fix  in  for grid visualization (heterogeneous array issue).

