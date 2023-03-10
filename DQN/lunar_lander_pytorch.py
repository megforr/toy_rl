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

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        """
        Forward pass through DQN. Can be called with single input or in batch.

        :param x: state
        :return: tensor([[left0exp, right0exp]...]).
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class Agent(object):
    """
    Agent code to select actions
    """
    def __init__(self, eps_start, eps_end, eps_decay):
        self.steps_done = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def select_action(self, state, env, policy_net):
        """
        Decayed epsilon greedy policy to select actions
        :param state: current state observation
        :param env: gym environment
        :param policy_net: current trained network
        :return: selected action
        """
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1 * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net(state).max(1)[1].view(1,1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def plot_durations(show_result=False):
    plt.figure(1)
    #plot_val_t = torch.tensor(episode_durations, dtype=torch.float)
    plot_val_t = torch.tensor(episodes_reward, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    #plt.ylabel('Duration')
    plt.ylabel('Reward')
    plt.plot(plot_val_t.numpy())
    # Take 100 episode averages and plot them too
    if len(plot_val_t) >= 100:
        means = plot_val_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


if __name__ == '__main__':

    #env = gym.make('LunarLander-v2', render_mode='human')
    env = gym.make('LunarLander-v2')

    BATCH_SIZE = 128            # Sample size from replay buffer
    GAMMA = 0.99                # Discount factor of returns
    EPS_START = 0.9             # Start value of epsilon
    EPS_END = 0.05              # Min value of epsilon
    EPS_DECAY = 1000            # Controls rate of epsilon decay, higher means slower decay
    TAU = 0.005                 # Update rate of the target network
    LR = 1e-4                   # Learning rate of the ADAM optimizer

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state, info = env.reset(seed=0)
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    agent = Agent(eps_start=EPS_START,
                  eps_end=EPS_END,
                  eps_decay=EPS_DECAY)

    episode_durations = []
    episodes_reward = []

    max_episodes = 500

    for episode_idx in range(max_episodes):

        episode_reward = 0.0

        if episode_idx % 10 == 0:
            print(episode_idx)
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            action = agent.select_action(state, env, policy_net)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            episode_reward += reward
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            #print(action, observation, reward, terminated, truncated, reward, done)

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.store(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # ????? ??? ?? ?? + (1 ????? )?????
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                episodes_reward.append(episode_reward)
                plot_durations()
                break

    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()