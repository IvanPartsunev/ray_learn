import gymnasium as gym

from ray_core_example.environment import Environment


class GymEnvironment(Environment, gym.Env):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

gym_env = GymEnvironment()
