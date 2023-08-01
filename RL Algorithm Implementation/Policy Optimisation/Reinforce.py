# A Monte-Carlo policy gradient algorithm
'''
每跑完一个episode,就用得到的trajectory (like MC) 进行一次network traning,直到reward值满足要求threshold.
用policy生成network作为policy approximator, selec_action通过对当前状态下,网络输出动作的概率分布采样,得到动作
finis_episode根据策略梯度损失函数对网络进行更新/训练
'''

import argparse
import gym
import numpy as np
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class Policy(nn.Module):  # Policy Approximate with parameter theta to be optimised
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)  # 是输入层到隐藏层的线性变换（全连接层），输入维度是4，隐藏层维度是128
        self.dropout = nn.Dropout(p=0.6)  # 是一个Dropout层，用于随机地将神经元的输出置为0，以防止过拟合
        self.affine2 = nn.Linear(128, 2)  # 是隐藏层到输出层的线性变换，输出维度是2，对应两个动作的分数值

        self.saved_log_probs = []  # 保存每个动作的对数概率值，用于计算损失函数。
        self.rewards = []  # 用于保存每个时间步的奖励值

    def forward(self, x):  # 模型的前向传播计算流程
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)  # 将动作分数值通过F.softmax函数进行Softmax操作，得到动作的概率分布。


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)  # 将状态转换为PyTorch的张量，并进行一些预处理，例如增加一个维度以适应模型输入的要求
    probs = policy.forward(state)  # 将状态输入策略网络，获得动作的概率分布
    m = Categorical(probs)  # 构造一个分类分布
    action = m.sample()  # 根据概率分布probs对动作进行采样，得到一个动作action
    policy.saved_log_probs.append(m.log_prob(action)) # 将选择的动作对数概率m.log_prob(action)保存到policy.saved_log_probs中
    return action.item()  # 并返回动作的整数值


def finish_episode(): # 对策略网络进行一次参数更新，使用之前采集到的动作和奖励信息计算策略梯度损失，并执行梯度上升优化算法
    R = 0
    policy_loss = []
    returns = []

    '''Use temporal causality: calculate return from t timestep'''
    # 从最后一个时间步开始，依次计算每个时间步的折扣回报，并保存在returns列表中。
    for r in policy.rewards[::-1]:
        R = r + args["gamma"] * R # R是反着计算出来的之后的reward总和,r是当前timestep的reward
        returns.insert(0, R)

    # 对returns进行归一化处理，减去均值并除以标准差，以稳定训练
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    # 使用对数概率和折扣回报计算策略梯度损失函数policy_loss
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)

    # Training
    optimizer.zero_grad() # 将之前的梯度信息清零。在每个训练循环之前，需要将之前的梯度清除，否则梯度会累积。
    policy_loss = torch.cat(policy_loss).sum()
    # torch.cat()将这些损失值连接成一个张量（tensor）
    # 然后使用.sum()函数对张量中的所有元素求和，得到最终的损失值policy_loss
    policy_loss.backward() # 反向传播
    optimizer.step() # 梯度下降
    # 在反向传播后，梯度信息已经计算好了，通过调用优化器（如SGD、Adam、RMSprop等）的.step()函数，可以根据梯度信息更新模型的参数

    del policy.rewards[:]
    del policy.saved_log_probs[:]
    # del语句用于从内存中删除这些列表中的所有元素，即清空列表。


args = {
    "gamma": 0.99,
    "seed": 543,
    "render": False,
    "log_interval": 10
}

running_reward = 10
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()

for i_episode in count(1):
    state, ep_reward = env.reset(seed=args["seed"]), 0
    state = state[0]
    for t in range(1, 10000):  # Don't infinite loop while learning policy, optimise parameter theta
        # 循环来完成一个episode,得到一个trajectory
        action = select_action(state)  # 从概率分布进行采样来选择动作
        state, reward, done, _, _ = env.step(action)
        policy.rewards.append(reward)
        ep_reward += reward
        if done:
            break

    running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
    # 通过对之前若干回合的回报进行平均计算得到的。每当智能体完成一个新的回合，就将该回合的累计回报与之前的平均回报进行加权平均
    # alpha是一个取值范围在0到1之间的超参数，用于控制平均的权重。一般情况下，alpha会设置成一个接近于1的值，
    # 这样可以让最近的回合对平均回报的影响更大，使得"running reward"能够更快地反映出策略的改善。
    # 它是对累计回报的一种平均化处理，用于更稳定地评估策略的性能。
    finish_episode()

    if i_episode % args["log_interval"] == 0:
        print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
            i_episode, ep_reward, running_reward))
    # 如果滑动平均回报running_reward超过了环境的奖励阈值，表示问题已经解决。
    if running_reward > env.spec.reward_threshold:
        print("Solved! Running reward is now {} and "
              "the last episode runs to {} time steps!".format(running_reward, t))
        break
