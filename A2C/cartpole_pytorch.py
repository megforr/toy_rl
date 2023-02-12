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
import math

# this ensures that the current MacOS version is at least 12.3+
print(torch.backends.mps.is_available())
# this ensures that the current current PyTorch installation was built with MPS activated.
print(torch.backends.mps.is_built())
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # Need 12.3+ for mps

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

# Experience replay buffer
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """
    Experience replay buffer to store transitions that can be sampled iid to train the DQN agent
    """
    def __init__(self, max_capacity):
        self.memory = deque([], maxlen=max_capacity)

    def store(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class A2C(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(A2C, self).__init__() # check to see if actor critic needs a different super()
        self.common1 = nn.Linear(n_observations, 128)
        #self.common2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, n_actions)  # policy head
        self.critic = nn.Linear(128, 1)         # value head

    def forward(self, x):
        """
        Forward pass through A2C. Can be called with single input or in batch.
        :param x: state
        :return: policy net, value net
        """

        x = F.relu(self.common1(x))
        #x = F.relu(self.common2(x))
        return self.actor(x), self.critic(x)



if __name__ == '__main__':

    env = query_environment('CartPole-v1')




