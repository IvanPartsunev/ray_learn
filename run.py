import ray
from environment import Environment
from evaluate_policy import evaluate_policy
from train import train_policy_parallel

ray.init()

num_episodes=10000
environment = Environment()


parallel_policy = train_policy_parallel(environment, num_episodes=num_episodes)
evaluate_policy(environment, parallel_policy, num_episodes=num_episodes)