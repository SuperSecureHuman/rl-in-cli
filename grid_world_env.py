import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import os
from gymnasium.utils import seeding


class GridWorldEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self,
                 grid_width=10,
                 grid_height=10,
                 player_start_pos=None,
                 render_mode=None):
        super(GridWorldEnv, self).__init__()
        self.grid_width = grid_width
        self.grid_height = grid_height
        # Define action space: N, NE, E, SE, S, SW, W, NW
        self.action_space = spaces.Discrete(8)
        # Observation space: [player_x, player_y, block_x, block_y]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array(
                [
                    self.grid_width - 1,
                    self.grid_height - 1,
                    self.grid_width - 1,
                    self.grid_height - 1,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        self.player_start_pos = player_start_pos
        self.render_mode = render_mode
        self.max_steps = 200
        self.current_step = 0

        self.np_random = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, seed = seeding.np_random(seed)
        self._seed = seed

        if self.player_start_pos is not None:
            self.player_pos = np.array(self.player_start_pos, dtype=np.int32)
        else:
            # Default to center of the grid
            self.player_pos = np.array(
                [self.grid_width // 2, self.grid_height // 2], dtype=np.int32)
        # Block is placed randomly in the grid
        self.block_pos = self.np_random.integers(
            low=[0, 0],
            high=[self.grid_width, self.grid_height],
            size=(2, ),
            dtype=np.int32,
        )
        while np.array_equal(self.player_pos, self.block_pos):
            self.block_pos = self.np_random.integers(
                low=[0, 0],
                high=[self.grid_width, self.grid_height],
                size=(2, ),
                dtype=np.int32,
            )
        self.steps_taken = 0
        self.current_step = 0
        self.prev_distance = self._compute_distance()
        return self._get_obs(), {}

    def step(self, action):
        action = int(action)

        self.steps_taken += 1
        self.current_step += 1
        moves = {
            0: np.array([0, 1]),  # North
            1: np.array([1, 1]),  # Northeast
            2: np.array([1, 0]),  # East
            3: np.array([1, -1]),  # Southeast
            4: np.array([0, -1]),  # South
            5: np.array([-1, -1]),  # Southwest
            6: np.array([-1, 0]),  # West
            7: np.array([-1, 1]),  # Northwest
        }
        move = moves.get(action, np.array([0, 0]))
        self.player_pos += move
        # Ensure the player stays within grid boundaries
        self.player_pos[0] = np.clip(self.player_pos[0], 0,
                                     self.grid_width - 1)
        self.player_pos[1] = np.clip(self.player_pos[1], 0,
                                     self.grid_height - 1)
        distance = self._compute_distance()
        terminated = np.array_equal(self.player_pos, self.block_pos)
        truncated = self.current_step >= self.max_steps
        # Reward is negative distance to the block
        reward = -float(distance)
        reward -= 0.1  # Small penalty for each step taken
        # Set info dictionary
        info = {'distance': distance}
        if self.render_mode == 'human':
            self.render()
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'human':
            # Clear the console
            if os.name == 'nt':
                os.system('cls')
            else:
                print('\033[H\033[J', end='')
            grid = np.full((self.grid_height, self.grid_width), '_', dtype=str)
            grid[self.block_pos[1], self.block_pos[0]] = 'B'
            grid[self.player_pos[1], self.player_pos[0]] = 'P'
            print('\n'.join(' '.join(row) for row in grid[::-1]))
            print(
                f"Player Position: {self.player_pos}, Block Position: {self.block_pos}"
            )
            print(f"Steps Taken: {self.steps_taken}")
            # Add a slight delay to make the rendering visible
            time.sleep(0.25)
        else:
            pass  # Other render modes

    def close(self):
        pass

    def _get_obs(self):
        return np.concatenate(
            (self.player_pos, self.block_pos)).astype(np.float32)

    def _compute_distance(self):
        # Use Manhattan distance
        return np.sum(np.abs(self.player_pos - self.block_pos))
