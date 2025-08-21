import ray
import time

from environment import Environment
from policy import Policy, PolicyStore


class Simulation:
  def __init__(self, env: Environment):
    """Simulates rollouts of an environment, given a policy to follow."""
    self.env = env

  def rollout(self, policy: Policy, render=False, explore=True, epsilon=0.1):
    """Returns experiences for a policy rollout."""

    experience = []
    state = self.env.reset()
    done = False
    while not done:
      action = policy.get_action(state, explore, epsilon)
      next_state, reward, done, info = self.env.step(action)
      experience.append([state, action, reward, next_state])
      state = next_state
      if render:
        time.sleep(0.05)
        self.env.render()
    return experience


@ray.remote
class SimulationActor(Simulation):
  """Ray actor for a Simulation."""
  ...