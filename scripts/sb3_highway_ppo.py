import gym
import torch as th
from stable_baselines3 import PPO
from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
import highway_env
import time

# ==================================
#        Main script
# ==================================

if __name__ == "__main__":
    train = True
    if train:
        n_cpu = 32
        batch_size = 64
        env = make_vec_env("merge-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        model = PPO("MlpPolicy",
                    env,
                    policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                    n_steps=24,
                    batch_size=batch_size * 12 // n_cpu,
                    n_epochs=10,
                    learning_rate=5e-4,
                    gamma=0.8,
                    verbose=2,
                    tensorboard_log="highway_ppo/")
        # Train the agent
        # model_loaded = PPO.load("/content/drive/MyDrive/researchHub/highway-env/highway_ppo/model")
        # model.set_parameters(model_loaded.get_parameters())
        model.learn(total_timesteps=int(3e6))
        # Save the agent
        model.save("highway_ppo/model")

    model = PPO.load("scripts/highway_ppo/model", device="cpu")
    env = gym.make("merge-v0")
    env.configure({"offscreen_rendering": False})
    img = env.render(mode='rgb_array')
    while True:
        obs = env.reset()
        done = [False]
        while not all(done):
            action, _ = model.predict(obs)
            # obs, reward, done, info = env.step(np.array([0,0],dtype=np.float32))
            obs, reward, done, info = env.step(tuple([a for a in action]))
            print(obs)
            print(done)
            env.render()
            time.sleep(0.3)
        env.close()
