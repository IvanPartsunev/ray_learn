import random
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Discrete

ACTIONS = {
  0: (1, 0),
  1: (0, -1),
  2: (-1, 0),
  3: (0, 1),
}

# class Discrete:
#   def __init__(self, num_actions: int):
#     self.n = num_actions
#
#   def sample(self):
#     return random.randint(0, self.n - 1)


class Environment:
  def __init__(self, config):
    self.size = config.get("size", 5)
    self.actions = config.get("actions", 4)
    self.start_x = config.get("start_x", 0)
    self.start_y = config.get("start_y", self.start_x)
    self.start = (self.start_x, self.start_y)
    self.end = (self.size - 1, self.size - 1)
    self.seeker, self.goal = self.start, self.end
    self.info = {"seeker": self.seeker, "goal": self.goal}

    self.action_space = Discrete(self.actions)
    self.observation_space = spaces.Box(
      low=0, high=1, shape=(self.size * self.size,), dtype=np.float32
    )

  def get_observation(self):
    obs = np.zeros(self.size * self.size, dtype=np.float32)
    idx = self.size * self.seeker[0] + self.seeker[1]
    obs[idx] = 1.0
    return obs

  def reset(self, seed, options):
    """Reset seeker position and return observations."""
    self.seeker = self.start
    # self.seeker = random.randrange(0, self.size), random.randrange(0, self.size)
    obs = self.get_observation()
    info = self.info
    return obs, info


  def get_reward(self) -> int:
    """Reward finding the goal"""
    return 1 if self.seeker == self.goal else 0

  def is_done(self) -> bool:
    """We're done if we found the goal"""
    return self.seeker == self.goal

  def step(self, action):
    """Take a step in a direction and return all available information."""

    move = ACTIONS.get(action, None)
    if not move:
      raise ValueError("Invalid action")

    new_position = self.seeker[0] + move[0], self.seeker[1] + move[1]
    valid = self.in_bound(new_position)
    if valid:
      self.seeker = new_position

    obs = self.get_observation()
    rew = self.get_reward()
    terminated = self.is_done()
    truncated = False
    return obs, rew, terminated, truncated, self.info

  def render(self, *args, **kwargs):
    """Render the environment, e.g., by printing its representation."""
    # Clear screen using ANSI escape sequences to avoid TERM dependency
    print("\033[2J\033[H", end="")

    grid = [["| " for _ in range(self.size * 1)] + ["|\n"] for _ in range(self.size)]
    grid[self.goal[0]][self.goal[1]] = "|G"
    grid[self.seeker[0]][self.seeker[1]] = "|S"
    print(''.join([''.join(grid_row) for grid_row in grid]))

  def in_bound(self, position: tuple[int, int]):
    x, y = position
    return 0 <= x < self.size and 0 <= y < self.size
