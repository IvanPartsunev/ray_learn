import random
import numpy as np
import ray

from environment import Environment


class Policy:
  def __init__(self, env: Environment):
    """
    A Policy suggests actions based on the current state.
    We do this by tracking the value of each state-action pair.
    """
    self.state_action_table = [
      [0 for _ in range(env.action_space.n)]
      for _ in range(env.observation_space.n)
    ]
    self.action_space = env.action_space

  def get_action(self, state, explore=True, epsilon=0.1):
    """
    Explore randomly or exploit the best value currently available.
    """
    if explore and random.uniform(0, 1) < epsilon:
      return self.action_space.sample()
    return np.argmax(self.state_action_table[state])


@ray.remote
class PolicyStore:
  def __init__(self):
    self.policy = None

  def put_policy(self, policy: Policy):
    self.policy = policy

  def get_policy(self):
    return self.policy

  def update_policy(self, experience: list, weight=0.1, discount_factor=0.9):
    """
    Updates a given policy with a list of (state, action, reward, state)
    experiences.
    """
    for state, action, reward, next_state in experience:
      next_max = np.max(self.policy.state_action_table[next_state])
      value = self.policy.state_action_table[state][action]
      new_value = (1 - weight) * value + weight * (reward + discount_factor * next_max)
      self.policy.state_action_table[state][action] = new_value
