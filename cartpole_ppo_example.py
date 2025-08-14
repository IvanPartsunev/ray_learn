import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig

# Start Ray
ray.init(ignore_reinit_error=True, num_cpus=1)

# Configure and run PPO on CartPole-v1
tune.run(
    "PPO",
    stop={"episode_reward_mean": 150},
    config={
        "env": "CartPole-v1",
        "framework": "tf", # or "tf" if you have TensorFlow installed
    },
    verbose=1,
)