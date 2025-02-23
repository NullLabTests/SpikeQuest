import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
import unittest

class SNN:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.layers = [10, 20, 15, 4]  # Neurons: [input, hidden1, hidden2, output]
        self.weights = [np.random.rand(self.layers[i], self.layers[i+1]) * 0.1 
                        for i in range(len(self.layers)-1)]
        self.potentials = [np.zeros(n) for n in self.layers]
        self.threshold = 1.0
        self.visited = set()
        self.obstacles = [(2, 2)]  # Example obstacle

    def forward(self, input_spikes):
        self.potentials[0] = input_spikes
        for l in range(len(self.layers)-1):
            # Integrate inputs
            self.potentials[l+1] = np.dot(self.potentials[l], self.weights[l])
            # Lateral inhibition (reduce neighbors by 0.2 if a neuron fires)
            winners = self.potentials[l+1] > self.threshold
            if np.any(winners):
                self.potentials[l+1] -= 0.2 * winners * (1 - np.eye(len(winners)))
            # Normalize (prevent runaway activations)
            total = np.sum(self.potentials[l+1])
            if total > 0:
                self.potentials[l+1] /= total
            # Decay
            self.potentials[l+1] *= 0.9

        return self.potentials[-1] > self.threshold

    def update_weights(self, spikes, reward):
        for l in range(len(self.weights)):
            pre = spikes[l]
            post = spikes[l+1]
            for i in range(len(pre)):
                for j in range(len(post)):
                    if post[j] and pre[i]:  # STDP
                        delta = 0.01 * reward if post[j] > pre[i] else -0.01 * reward
                        self.weights[l][i, j] = np.clip(self.weights[l][i, j] + delta, 0, 1)

    def is_valid_move(self, pos):
        x, y = pos
        return (0 <= x < self.grid_size and 0 <= y < self.grid_size and pos not in self.obstacles)

    def move(self, pos, action):
        x, y = pos
        if action == 0 and self.is_valid_move((x, y + 1)): y += 1  # Up
        elif action == 1 and self.is_valid_move((x, y - 1)): y -= 1  # Down
        elif action == 2 and self.is_valid_move((x - 1, y)): x -= 1  # Left
        elif action == 3 and self.is_valid_move((x + 1, y)): x += 1  # Right
        return (x, y)

    def step(self, pos):
        x, y = pos
        input_spikes = np.zeros(10)
        input_spikes[x] = 1  # Encode x
        input_spikes[5 + y] = 1  # Encode y
        spikes = [input_spikes]
        output = self.forward(input_spikes)
        spikes.append(self.potentials[1] > self.threshold)
        spikes.append(self.potentials[2] > self.threshold)
        spikes.append(output)

        action = np.argmax(output) if np.any(output) else np.random.randint(4)
        new_pos = self.move(pos, action)
        reward = 1 if new_pos == (self.grid_size - 1, self.grid_size - 1) else \
                 (0.2 if new_pos not in self.visited else 0)
        self.visited.add(new_pos)
        self.update_weights(spikes, reward)
        return new_pos, reward

    def run_episode(self, max_steps=100):
        pos = (0, 0)
        path = [pos]
        rewards = []
        for t in range(max_steps):
            pos, reward = self.step(pos)
            path.append(pos)
            rewards.append(reward)
            if pos == (self.grid_size - 1, self.grid_size - 1):
                print(f"Goal reached at step {t+1}!")
                break
        return path, rewards

    def visualize_path(self, path):
        fig, ax = plt.subplots(figsize=(8, 8))
        grid = np.zeros((self.grid_size, self.grid_size))
        for x, y in self.obstacles:
            grid[x, y] = -1  # Obstacles
        grid[self.grid_size - 1, self.grid_size - 1] = 2  # Goal
        for x, y in path:
            if (x, y) not in self.obstacles:
                grid[x, y] = 1 if (x, y) not in self.visited else 0.5  # Visited vs. new

        cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap
        im = ax.imshow(grid, cmap=cmap, vmin=-1, vmax=2)
        ax.set_title("SpikeQuest: Agent Path in 5x5 Grid")
        ax.set_xticks(np.arange(self.grid_size))
        ax.set_yticks(np.arange(self.grid_size))
        ax.grid(True, which="both", linestyle='-', color='black')
        plt.colorbar(im, ax=ax, label='State: -1=Obstacle, 0.5=Visited, 1=New, 2=Goal')

        # Animate the path
        def update(frame):
            ax.clear()
            grid = np.zeros((self.grid_size, self.grid_size))
            for x, y in self.obstacles:
                grid[x, y] = -1
            grid[self.grid_size - 1, self.grid_size - 1] = 2
            for i, (x, y) in enumerate(path[:frame + 1]):
                grid[x, y] = 1 if (x, y) not in self.visited else 0.5
            im = ax.imshow(grid, cmap=cmap, vmin=-1, vmax=2)
            ax.set_title(f"Step {frame}")
            ax.set_xticks(np.arange(self.grid_size))
            ax.set_yticks(np.arange(self.grid_size))
            ax.grid(True, which="both", linestyle='-', color='black')
            return [im]

        anim = FuncAnimation(fig, update, frames=len(path), interval=500, blit=True)
        plt.show()

    def visualize_network(self, step, spikes):
        G = nx.DiGraph()
        for l in range(len(self.layers)-1):
            for i in range(self.layers[l]):
                for j in range(self.layers[l+1]):
                    G.add_edge(f"L{l}N{i}", f"L{l+1}N{j}", weight=self.weights[l][i, j])
        
        pos = nx.spring_layout(G)
        plt.figure(figsize=(12, 8))
        node_colors = ['blue' if not s else 'red' for s in np.any(spikes, axis=0)]
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, 
                edge_color='gray', width=0.5, font_size=8)
        plt.title(f"SpikeQuest SNN at Step {step}")
        plt.show()

# Automated Tests
class TestSpikeQuest(unittest.TestCase):
    def setUp(self):
        self.net = SNN(grid_size=5)

    def test_goal_reach(self):
        path, rewards = self.net.run_episode(max_steps=50)
        self.assertEqual(path[-1], (4, 4), "Agent should reach the goal (4,4)")
        self.assertLessEqual(len(path), 50, "Should reach goal in ≤ 50 steps")

    def test_path_diversity(self):
        paths = []
        for _ in range(5):  # Run 5 trials
            self.net = SNN(grid_size=5)  # Reset network
            path, _ = self.net.run_episode(max_steps=50)
            paths.append(path)
        unique_paths = len(set(tuple(p) for p in paths))
        self.assertGreater(unique_paths, 1, "Network should explore diverse paths")

    def test_stability(self):
        path, rewards = self.net.run_episode(max_steps=50)
        for p in self.net.potentials:
            self.assertLessEqual(np.max(p), 1.5, "Potentials should not exceed 1.5 (stability check)")

if __name__ == "__main__":
    # Run the SNN
    net = SNN(grid_size=5)
    path, rewards = net.run_episode()
    print(f"Path: {path}")
    print(f"Total Reward: {sum(rewards)}")

    # Visualize results
    net.visualize_path(path)
    spikes = [np.zeros(10), np.zeros(20), np.zeros(15), np.zeros(4)]  # Example spikes (replace with actual)
    net.visualize_network(0, spikes)

    # Run tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
