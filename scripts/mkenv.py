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
for _ in range(75):

    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()

plt.imshow(env.render(mode="rgb_array"))
plt.show()