import sys

import numpy as np
import ray

from environment import Environment
from policy import Policy
from simulation import Simulation, SimulationActor


def update_policy(policy: Policy, experience: list, weight=0.1, discount_factor=0.9):
  """
  Updates a given policy with a list of (state, action, reward, state)
  experiences.
  """
  for state, action, reward, next_state in experience:
    next_max = np.max(policy.state_action_table[next_state])
    value = policy.state_action_table[state][action]
    new_value = (1 - weight) * value + weight * (reward + discount_factor * next_max)
    policy.state_action_table[state][action] = new_value


def train_policy(env: Environment, num_episodes=1000, weight=0.1, discount_factor=0.9):
  """
  Training a policy by updating it with rollout experiences.
  """
  policy = Policy(env)
  sim = Simulation(env)

  for _ in range(num_episodes):
    experience = sim.rollout(policy)
    update_policy(policy, experience, weight, discount_factor)
  return policy


def train_policy_parallel(env, num_episodes=1000, num_simulations=4):
  """Parallel policy training function."""
  policy = Policy(env)
  simulations = [SimulationActor.remote() for _ in range(num_simulations)]

  policy_ref = ray.put(policy)

  for i in range(num_episodes):
    experiences = [sim.rollout.remote(policy_ref) for sim in simulations]

    while len(experiences) > 0:
      finished, experiences = ray.wait(experiences)
      for xp in ray.get(finished):
        update_policy(policy, xp)
    progress_bar(i + 1, num_episodes)

  return policy


def progress_bar(current, total):
  percent = int(100 * current / total)
  filled = int(50 * current / total)
  bar = '█' * filled + '░' * (50 - filled)
  sys.stdout.write(f'\r|{bar}| {percent}%')
  sys.stdout.flush()
  if current >= total:  # Changed == to >= to be safe
    print()