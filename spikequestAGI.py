import numpy as np
import matplotlib.pyplot as plt

class SNN:
    def __init__(self):
        self.layers = [10, 20, 15, 4]  # Neurons per layer
        self.weights = [np.random.rand(self.layers[i], self.layers[i+1]) * 0.1 
                        for i in range(len(self.layers)-1)]
        self.potentials = [np.zeros(n) for n in self.layers]
        self.threshold = 1.0
        self.visited = set()

    def forward(self, input_spikes):
        self.potentials[0] = input_spikes
        for l in range(len(self.layers)-1):
            # Integrate inputs
            self.potentials[l+1] = np.dot(self.potentials[l], self.weights[l])
            # Lateral inhibition
            winners = self.potentials[l+1] > self.threshold
            if np.any(winners):
                self.potentials[l+1] -= 0.2 * winners * (1 - np.eye(len(winners)))
            # Normalize
            if np.sum(self.potentials[l+1]) > 0:
                self.potentials[l+1] /= np.sum(self.potentials[l+1])
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
        reward = 1 if new_pos == (4, 4) else (0.1 if new_pos not in self.visited else 0)
        self.visited.add(new_pos)
        self.update_weights(spikes, reward)
        return new_pos, reward

    def move(self, pos, action):
        x, y = pos
        if action == 0 and y < 4: y += 1  # Up
        elif action == 1 and y > 0: y -= 1  # Down
        elif action == 2 and x > 0: x -= 1  # Left
        elif action == 3 and x < 4: x += 1  # Right
        return (x, y)

# Run simulation
net = SNN()
pos = (0, 0)
path = [pos]
for t in range(50):
    pos, reward = net.step(pos)
    path.append(pos)
    if pos == (4, 4):
        print(f"Goal reached at step {t+1}!")
        break

# Plot path
x, y = zip(*path)
plt.plot(x, y, 'o-')
plt.grid(True)
plt.show()
