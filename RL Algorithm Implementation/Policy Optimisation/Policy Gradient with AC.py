'''
This is policy gradient with baseline and temporal causality and actor-critic.
Almost the same as the reinforce algorithm but with an extra baseline,introducing advantage function.
As it's actor-critic, it uses network to predict baseline and Gt
This requires a bit more modification towards the net (two heads),
and something shown in the "key part of the algorithm"
'''

import os
import argparse
import gym
import numpy as np
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

is_mps = torch.backends.mps.is_available()
print(is_mps)
print(torch.backends.mps.is_built())
mps = torch.device("mps")
print(mps)


def prepro(I):
    """ prepro = pre processing
    prepro 210x160x3 into 6400
    takes an input image I of shape 210x160x3 (height x width x channels)
    and preprocesses it into a 1D array of length 6400
    """
    I = I[35:195]  # crop the image
    # It keeps rows from index 35 to 194 (total of 195-35=160 rows) and keeps all the columns and channels.
    # This cropping is typically done to focus on the main game area, removing the score and other irrelevant parts of the image.
    I = I[::2, ::2, 0] # downsample the cropped image
    # It takes every second row and every second column, and only keeps the first channel (channel 0).
    # reduces the resolution of the image, which helps in reducing the computational cost and focusing on relevant information.
    I[I == 144] = 0
    I[I == 109] = 0
    # Replace all the pixels with R-value 144 and 109 (RGB values in the original image) with 0.
    #  filter out specific colors or elements from the image that are not essential for the agent's decision-making process.
    I[I != 0 ] = 1
    # Set all the remaining non-zero pixels in the downsampled image to 1
    # converts the preprocessed image into a binary image
    return I.astype(np.float64).ravel()


class PGbaseline(nn.Module):
    def __init__(self, num_actions=2):
        super(PGbaseline, self).__init__()
        self.affine1 = nn.Linear(6400, 200)  # preprocessing gives 1D array with length 6400
        self.action_head = nn.Linear(200, num_actions) # action 1: static, action 2: move up, action 3: move down
        self.value_head = nn.Linear(200, 1)

        self.num_actions = num_actions
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values

    def select_action(self, x):
        x = Variable(torch.from_numpy(x).float().unsqueeze(0))
        if is_mps: x = x.to(mps)
        probs, state_value = self.forward(x)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append((m.log_prob(action), state_value))
        return action


def finish_episode():
    R = 0
    policy_loss = []
    value_loss = []
    rewards = []

    for r in policy.rewards[::-1]:
        R = r + args["gamma"] * R
        rewards.insert(0, R)

    # turn rewards to pytorch tensor and standardize
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
    if is_mps: rewards = rewards.to(mps)

    ''' Key part of the algorithm '''
    for (log_prob, value), reward in zip(policy.saved_log_probs, rewards):
        advantage = reward - value  # baseline is expected return, expected return is value function
        policy_loss.append(- log_prob * advantage)         # policy gradient
        value_loss.append(F.smooth_l1_loss(value, reward)) # Calculate target error
        # value represents the estimated state value for the current state. It is obtained from the value head of the neural network.
        # reward is the actual reward obtained after performing the action and reaching the next state.
        # F.smooth_l1_loss(value, reward) computes the smooth L1 loss between the estimated value value and the actual reward reward.
        # The F.smooth_l1_loss computes the smooth L1 loss between the estimated value value and the actual reward reward.
        # The smooth L1 loss is less sensitive to outliers and helps in stabilizing the training of the value function.

    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum()
    value_loss = torch.stack(value_loss).sum()
    loss = policy_loss + value_loss
    if is_mps:
        loss.to(mps)
    loss.backward()
    optimizer.step()

    # clean rewards and saved_actions
    del policy.rewards[:]
    del policy.saved_log_probs[:]


# Main loop
args = {
    "gamma": 0.99,
    "decay_rate": 0.99,
    "learning_rate": 1e-4,
    "batch_size": 20,
    "seed": 87,
    "test": False
}


D = 80 * 80
# test = args["test"]
# if test == True:
#     render = True
# else:
#     render = False

# built policy network
policy = PGbaseline()
if is_mps:
    policy.to(mps)

# check & load pretrain model
# if os.path.isfile('pgb_params.pkl'):
#     print('Load PGbaseline Network parametets ...')
#     if is_mps:
#         policy.load_state_dict(torch.load('pgb_params.pkl'))
#     else:
#         policy.load_state_dict(torch.load('pgb_params.pkl', map_location=lambda storage, loc: storage))

# construct a optimal function
optimizer = optim.RMSprop(policy.parameters(), lr=args["learning_rate"], weight_decay=args["decay_rate"])

env = gym.make('Pong-v4')
torch.manual_seed(args["seed"])
running_reward = None
reward_sum = 0
for i_episode in count(1):
    state = env.reset(seed=args["seed"])
    state = state[0]
    prev_x = None
    for t in range(10000):
        # if render: env.render()
        cur_x = prepro(state) # cur_x is current state
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x
        # calculate the difference between the current observation and previous observation
        # to capture the changes in the environment between consecutive time steps
        action = policy.select_action(x)
        action_env = action + 2
        # 2 is an offset that matches calculated action with action from environment
        # Example: 0 -> 2 means action 2 move up
        # action 1: static, action 2: move up, action 3: move down
        state, reward, done, _, _ = env.step(action_env)
        reward_sum += reward
        policy.rewards.append(reward)

        if done:
            # tracking log
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('Policy Gradient with Baseline ep %03d done. reward: %f. reward running mean: %f' % (i_episode, reward_sum, running_reward))
            reward_sum = 0
            break


    # use policy gradient update model weights
    if i_episode % args["batch_size"] == 0 and args["test"] == False:
        finish_episode()

    # Save model in every 50 episode
    if i_episode % 50 == 0 and args["test"] == False:
        print('ep %d: model saving...' % (i_episode))
        torch.save(policy.state_dict(), 'pgb_params.pkl')