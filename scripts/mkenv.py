import gym
import highway_env
import os
from matplotlib import pyplot as plt
import numpy as np
import pprint
# %matplotlib inline
#@markdown <h3>← 输入需要进入的项目文件夹
os.chdir('/content/drive/MyDrive/researchHub')
retval = os.getcwd()
print("当前工作目录为 : %s" % retval)
path = "highway-env" #@param {type: "string"}
os.chdir(path)
retval = os.getcwd()
print("目录修改成功 : %s" % retval)
# %pip install .

env = gym.make('merge-v0')
env.configure({"offscreen_rendering": True})
img = env.render(mode='rgb_array')
env.reset()
dim_local_o = 2
world_size = np.ones(dim_local_o)
local_obs = np.zeros(dim_local_o)
for _ in range(75):

    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    # wall_dists = np.array([[world_size - obs[1], world_size - obs[2]],
    #                         [obs[1], obs[2]]])  
    # # wall_angles = np.array([0, np.pi / 2, np.pi, 3 / 2 * np.pi]) - obs[4]
    # closest_wall = np.argmin(wall_dists, axis =  1)
    # local_obs[0] = wall_dists[closest_wall][0]
    # local_obs[1] = wall_dists[closest_wall][1]
    # pprint.pprint(obs)
    # data = np.array(obs)
    # pprint.pprint(data)
    # pprint.pprint(data[...,2])
    env.render()

plt.imshow(env.render(mode="rgb_array"))
plt.show()