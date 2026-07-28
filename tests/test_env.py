import numpy as np
from spikequest.env.grid_world import GridWorld


class TestGridWorld:
    def setup_method(self):
        self.env = GridWorld(size=5, max_steps=50,
                             obstacles=[(2, 2)])

    def test_reset(self):
        obs = self.env.reset()
        assert self.env.agent_pos == (0, 0)
        assert self.env.step_count == 0
        assert (0, 0) in self.env.visited

    def test_step_up(self):
        self.env.reset()
        obs, reward, done, info = self.env.step(0)  # up
        assert self.env.agent_pos == (0, 1)
        assert not done

    def test_step_down_blocked(self):
        self.env.reset()
        obs, reward, done, info = self.env.step(1)  # down (blocked by boundary)
        assert self.env.agent_pos == (0, 0)  # stays in place
        assert reward == self.env.reward_obstacle

    def test_goal_reached(self):
        self.env.reset()
        # Teleport to goal-adjacent position
        self.env.agent_pos = (4, 3)
        obs, reward, done, info = self.env.step(0)  # up to (4,4)
        assert self.env.agent_pos == (4, 4)
        assert done
        assert reward == self.env.reward_goal

    def test_obstacle_block(self):
        self.env.reset()
        self.env.agent_pos = (2, 1)
        obs, reward, done, info = self.env.step(0)  # up to (2,2) blocked
        assert self.env.agent_pos == (2, 1)
        assert reward == self.env.reward_obstacle

    def test_visited_state(self):
        self.env.reset()
        self.env.visited.add((0, 1))
        self.env.agent_pos = (0, 1)
        obs, reward, done, info = self.env.step(2)  # left to (0,1) -> wait no
        # Actually let's just move to (0,1) and check it gets visited reward
        # Reset and go to (0,1) then back to (0,0)
        self.env.reset()
        self.env.step(0)  # (0,1)
        obs, reward, done, info = self.env.step(1)  # (0,0) visited
        self.env.agent_pos = (0, 0)
        # Actually let me restructure this more carefully
        self.env.reset()
        self.env.step(0)  # up to (0,1), reward = step + visited bonus (new)
        self.env.agent_pos = (0, 1)  # manually set for simplicity (above step moved here)
        obs, reward, done, info = self.env.step(1)  # down back to (0,0)
        assert reward == self.env.reward_step + self.env.reward_visited

    def test_max_steps_truncation(self):
        tiny_env = GridWorld(size=3, max_steps=5)
        tiny_env.reset()
        for i in range(4):
            obs, r, done, info = tiny_env.step(3)  # go right
        obs, r, done, info = tiny_env.step(3)  # 5th step
        assert done

    def test_partial_obs_shape(self):
        env = GridWorld(size=5, partial_obs=True, vision_radius=1)
        obs = env.reset()
        assert len(obs) == 9  # 3x3 patch

    def test_default_obstacles(self):
        env = GridWorld(size=10)
        assert len(env.obstacles) > 0

    def test_render_grid(self):
        self.env.reset()
        text = self.env.render_grid()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_render_path(self):
        self.env.reset()
        path = [(0, 0), (0, 1), (1, 1)]
        text = self.env.render_grid(path)
        assert "0" in text
        assert "1" in text
        assert "2" in text