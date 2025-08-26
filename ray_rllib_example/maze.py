from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig

from ray_rllib_example.maze_gym_env import GymEnvironment

tune.register_env("MyMazeEnv", lambda cfg: GymEnvironment(cfg))
config = (DQNConfig()
          .environment(env="MyMazeEnv")
          .training(model={"fcnet_hiddens": [64, 64], "fcnet_activation": "relu"})
          .env_runners(num_env_runners=2, create_env_on_local_worker=True))

tuner = tune.Tuner(
    "DQN",
    param_space=config.to_dict(),
    run_config=tune.RunConfig(stop={"training_iteration": 10}),
)

results = tuner.fit()