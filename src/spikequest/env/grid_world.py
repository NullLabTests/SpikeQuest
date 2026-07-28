import numpy as np
from typing import Optional, Tuple, List, Set


class GridWorld:
    """GridWorld navigation environment with Gymnasium-like interface.

    The agent starts at (0, 0) and must navigate to (N-1, N-1) while
    avoiding obstacles. Supports full observability (agent position)
    and partial observability (local egocentric view).

    Action space: {0: up, 1: down, 2: left, 3: right}
    Observation space depends on `partial_obs` flag.

    Args:
        size: grid side length
        max_steps: episode truncation length
        obstacles: list of (x, y) obstacle coordinates.
            If None, generates a default obstacle set.
        partial_obs: if True, observation is a local view patch
        vision_radius: radius of local view (only if partial_obs)
        reward_goal: reward for reaching goal (episode end)
        reward_step: per-step penalty/shaping
        reward_obstacle: penalty for bumping into obstacle
        reward_visited: penalty for re-visiting a state
    """

    def __init__(
        self,
        size: int = 10,
        max_steps: int = 200,
        obstacles: Optional[List[Tuple[int, int]]] = None,
        partial_obs: bool = False,
        vision_radius: int = 2,
        reward_goal: float = 10.0,
        reward_step: float = 0.0,
        reward_obstacle: float = -0.1,
        reward_visited: float = 0.0,
    ):
        self.size = size
        self.max_steps = max_steps
        self.partial_obs = partial_obs
        self.vision_radius = vision_radius
        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.reward_obstacle = reward_obstacle
        self.reward_visited = reward_visited

        if obstacles is not None:
            self.obstacles = set(obstacles)
        else:
            self.obstacles = self._default_obstacles()

        self.agent_pos: Optional[Tuple[int, int]] = None
        self.goal_pos: Tuple[int, int] = (size - 1, size - 1)
        self.step_count: int = 0
        self.visited: Set[Tuple[int, int]] = set()
        self.rng = np.random.RandomState()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.agent_pos = (0, 0)
        self.step_count = 0
        self.visited = {self.agent_pos}
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert 0 <= action <= 3, f"Invalid action {action}"
        self.step_count += 1

        dx, dy = [(0, 1), (0, -1), (-1, 0), (1, 0)][action]
        x, y = self.agent_pos
        nx, ny = x + dx, y + dy
        candidate = (nx, ny)

        if self._is_blocked(candidate):
            candidate = self.agent_pos
            reward = self.reward_obstacle
        else:
            if candidate == self.goal_pos:
                reward = self.reward_goal
            elif candidate not in self.visited:
                reward = self.reward_step + 0.0  # placeholder for novelty bonus
            else:
                reward = self.reward_step + self.reward_visited
            self.agent_pos = candidate
            self.visited.add(candidate)

        done = (candidate == self.goal_pos) or (self.step_count >= self.max_steps)
        return self._get_obs(), reward, done, {"steps": self.step_count}

    def _get_obs(self) -> np.ndarray:
        if self.partial_obs:
            return self._local_view()
        return np.array(self.agent_pos, dtype=np.float32)

    def _local_view(self) -> np.ndarray:
        r = self.vision_radius
        patch_size = 2 * r + 1
        view = np.zeros((patch_size, patch_size), dtype=np.float32)
        x0, y0 = self.agent_pos
        for i in range(patch_size):
            for j in range(patch_size):
                wx = x0 + (i - r)
                wy = y0 + (j - r)
                if wx < 0 or wx >= self.size or wy < 0 or wy >= self.size:
                    view[i, j] = -1.0
                elif (wx, wy) in self.obstacles:
                    view[i, j] = -0.5
                elif (wx, wy) == self.goal_pos:
                    view[i, j] = 1.0
                else:
                    view[i, j] = 0.0
        return view.flatten()

    def _is_blocked(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return True
        return pos in self.obstacles

    def _default_obstacles(self) -> Set[Tuple[int, int]]:
        n = self.size
        obs = set()
        if n == 10:
            for y in range(2, 5):
                obs.add((3, y))
            for y in range(3, 7):
                obs.add((6, y))
            for x in range(3, 7):
                obs.add((x, 6))
            obs.add((7, 2))
            obs.add((7, 3))
        else:
            mid = n // 2
            for i in range(1, n - 2):
                obs.add((mid, i))
        return obs

    def get_obs_dim(self) -> int:
        if self.partial_obs:
            r = self.vision_radius
            return (2 * r + 1) ** 2
        return 2

    def render_grid(self, path: Optional[List[Tuple[int, int]]] = None):
        grid = np.full((self.size, self.size), ".", dtype=str)
        for ox, oy in self.obstacles:
            grid[ox, oy] = "#"
        gx, gy = self.goal_pos
        grid[gx, gy] = "G"
        if path:
            for i, (px, py) in enumerate(path):
                if (px, py) not in self.obstacles and (px, py) != self.goal_pos:
                    grid[px, py] = str(i % 10)
        lines = []
        for y in range(self.size - 1, -1, -1):
            lines.append(" ".join(grid[x, y] for x in range(self.size)))
        return "\n".join(lines)