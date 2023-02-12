#!/usr/bin/env python

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import math


if __name__ == '__main__':

    env = gym.make('CartPole-v1')


    #https://towardsdatascience.com/installing-pytorch-on-apple-m1-chip-with-gpu-acceleration-3351dc44d67c
    # this ensures that the current MacOS version is at least 12.3+
    print(torch.backends.mps.is_available())
    # this ensures that the current current PyTorch installation was built with MPS activated.
    print(torch.backends.mps.is_built())

    dtype = torch.float
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # Need 12.3+ for mps

    max_episodes = 1

    for ep in range(max_episodes):

        obs, info = env.reset()
        state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


