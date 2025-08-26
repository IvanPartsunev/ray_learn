from simulation import Simulation


def evaluate_policy(env, policy, num_episodes = 1000, render=False):
  """Evaluate a trained policy through rollouts."""
  simulation = Simulation(env)
  steps = 0
  for i in range(num_episodes):
    experience = simulation.rollout(policy, render)
    steps += len(experience)

  print(f"{steps / num_episodes} steps on average "
        f"for a total of {num_episodes} episodes.")

  return steps / num_episodes


