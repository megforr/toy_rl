#!/usr/bin/env python

import numpy as np
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# this ensures that the current MacOS version is at least 12.3+
print(torch.backends.mps.is_available())
# this ensures that the current current PyTorch installation was built with MPS activated.
print(torch.backends.mps.is_built())
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # Need 12.3+ for mps

class REINFORCEPGN(nn.Module):

    def __init__(self, input_size, n_actions):
        super(REINFORCEPGN, self).__init__()

        self.net = nn.Sequential(nn.Linear(input_size, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, n_actions)
                                 )

    def forward(self, x):
        return self.net(x)

class Agent(object):

    def __init__(self):



def calc_qvalues(rewards):
    """
    Takes a list of rewards for whole episode and calculates
    the discounted total reward for each step.

    Discounted total reward for each step can be used to calculate
    the loss.

    :param rewards: list of episode rewards
    :return: discounted reward for each step
    """
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))

def query_environment(name):
    env = gym.make(name)
    spec = gym.spec(name)
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Max Episode Steps: {spec.max_episode_steps}")
    print(f"Nondeterministic: {spec.nondeterministic}")
    print(f"Reward Range: {env.reward_range}")
    print(f"Reward Threshold: {spec.reward_threshold}")

    return env


if __name__ == '__main__':

    GAMMA = 0.99
    LEARNING_RATE = 0.01
    EPISODES_TO_TRAIN = 4

    MAX_EPISODES = 200

    env = query_environment('CartPole-v1')

    net = REINFORCEPGN(env.observation_space.shape[0],
                       env.action_space.n)

    #for episode_idx in range(MAX_EPISODES):








