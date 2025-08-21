import ray
from environment import Environment, Discrete
from evaluate_policy import evaluate_policy
from policy import Policy
from simulation import Simulation
from train import train_policy_parallel, train_policy

# ray.init()
#
num_episodes=100000
environment = Environment(size=20)
#
#
parallel_policy = train_policy_parallel(environment, num_episodes=num_episodes)
# evaluate_policy(environment, parallel_policy, num_episodes=num_episodes)

# import time
# environment = Environment()
# while not environment.is_done():
#   random_action = environment.action_space.sample()
#   environment.step(random_action)
#   time.sleep(0.1)
#   environment.render()

# untrained_policy = Policy(environment)
# sim = Simulation(environment)
# exp = sim.rollout(untrained_policy, render=False, epsilon=1.0)
# print(f"{'=' * 10}\nExperience\n{'=' * 10}")
# for exp_row in exp:
#   print(exp_row)
# print()
# print(f"{'=' * 10}\nPolicy\n{'=' * 10}")
# for row in untrained_policy.state_action_table:
#   print(row)

# trained_policy = train_policy(environment)
# print(f"{'=' * 10}\nPolicy\n{'=' * 10}")
# for row in trained_policy.state_action_table:
#   print(row)