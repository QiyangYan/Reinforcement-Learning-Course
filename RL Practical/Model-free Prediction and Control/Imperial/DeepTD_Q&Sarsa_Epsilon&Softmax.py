''' Readme
This code contains the conde for two temporal difference algorithms: Q-learning(off policy) and SARSA(on policy).
They are constructed based on a neural network and being tested with two policies: Epsilon-greedy and Softmax.
The difference in performance between these algorithms can be found by setting: training = 1.
This script will store the trained weight as a name.h5 function, which can be further tested by setting: training = 0.
Remember to set the policy indicator to the policy you want to use for the training: "epsilon" or "softmax"
'''


import random
import gym
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.activations import softmax
from keras.optimizers.legacy import Adam
import os
import time
import numpy as np
import matplotlib.pyplot as plt


def softmax(x, temperature=0.025):
    # x is the score, the action value function of states
    """Compute softmax values for each sets of scores in x."""
    x = (x - np.expand_dims(np.max(x, 1), 1))
    # 这一步将每一行的元素减去该行中的最大值，用于数值稳定性，避免指数函数的输入过大而导致数值溢出或不稳定。
    # np.max(array, axis): 沿着指定的轴（axis）计算数组 array 中的最大值。例如，axis=0 表示沿着列方向计算最大值，axis=1 表示沿着行方向计算最大值。
    # np.expand_dims()在数组中插入新的维度,将得到的最大值数组转换为一个“行向量”。
    # 例如，如果 x 是一个形状为 (m, n) 的二维数组，那么 np.max(x, 1) 的结果将是一个形状为 (m,) 的一维数组。
    # 通过 np.expand_dims() 函数将其转换为形状为 (m, 1) 的二维数组，其中每个元素是对应行的最大值。
    x = x / temperature
    # 这一步将 x 中的每个元素除以 temperature，以调整概率分布的平滑程度
    e_x = np.exp(x)
    return e_x / (np.expand_dims(e_x.sum(1), -1) + 1e-5)
    # e_x.sum(1) 的作用是用于计算每一行元素的指数形式的和, 得到一行一维数组
    # np.expand_dims()在数组中插入新的维度,将得到的最大值数组转换为一个“列向量”,这样可以和很多行除。
    # 除以各行元素的和, Normalise 指数概率分布为概率值(分布)
    # 为了避免分母为零，很小的数值（如 1e-5）被添加到分母中。这样可以确保分母不会为零，从而避免了数值计算上的不稳定性和错误。


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model
        # 把state作为输入,来预测action value function Q

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        # this is the memory that stores transition tuples that's sampled from dataset

    def act(self, state, policyIndicator):
        # ---- the epsilon-greedy policy ----
        if policyIndicator == 0:
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])  # returns action value function
        # ---- softmax ----
        else:
            act_values = self.model.predict(state)
            probabilities = softmax(act_values)
            action = np.random.choice(np.arange(self.action_size), p=probabilities[0])
            # np.random.choice 函数根据概率分布 probabilities[0] 随机选择一个动作。
            # self.action_size 表示动作空间的大小
            # np.arange(self.action_size) 生成了一个包含从 0 到 self.action_size-1 的整数数组，用于表示动作的索引。
            return action

    def exploit(self, state):  # When we test the agent we dont want it to explore anymore, but to exploit what it has learnt
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size, indicator, policy_indicator):
        minibatch = random.sample(self.memory, batch_size)

        state_b = np.squeeze(np.array(list(map(lambda x: x[0], minibatch))))
        action_b = np.squeeze(np.array(list(map(lambda x: x[1], minibatch))))
        reward_b = np.squeeze(np.array(list(map(lambda x: x[2], minibatch))))
        next_state_b = np.squeeze(np.array(list(map(lambda x: x[3], minibatch))))
        done_b = np.squeeze(np.array(list(map(lambda x: x[4], minibatch))))

        if indicator == 0:
            ### Q-learning
            target = (reward_b + self.gamma * np.amax(self.model.predict(next_state_b), 1))  # target value
            target[done_b == 1] = reward_b[done_b == 1] # 如果当前状态 done_b 为 1（即终止状态），那么将 target 的值更新为 reward_b。
            # 这是因为在终止状态时，我们没有后续的动作和状态，因此 Q-value 的更新应该直接使用当前状态的即时奖励。
            target_f = self.model.predict(state_b)

        else:
            ### SARSA
            # S, A, R, S'
            next_q = self.model.predict(next_state_b)
            next_a = self.act(next_state_b, policy_indicator)  # find action-index use epsilon-greedy
            target = reward_b + self.gamma * next_q[range(batch_size), next_a]  # find target value
            target[done_b == 1] = reward_b[done_b == 1]
            target_f = self.model.predict(state_b)

        for k in range(target_f.shape[0]):  # update use Net
            target_f[k][action_b[k]] = target[k]
            # target_f[k][action_b[k]] 表示神经网络对当前状态下执行的动作的 Q-value 预测
            # kth state, kth action_b, 's Q_value

        self.model.train_on_batch(state_b, target_f)
        # 函数已经帮减去当前 Q-value, target error is calculated by the function
        # 特征向量在神经网络中的加权过程是由神经网络的权重参数来决定的，而不需要在代码中显式地乘以任何系数
        # train_on_batch() 方法会使用输入数据 x 和目标数据 y 来计算模型的损失，并根据指定的优化器更新模型的参数，从而最小化损失函数。

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def q_learning_epsilon(policy):
    if policy == "softmax":
        policyIndicator = 1
    else:
        policyIndicator = 0

    EPISODES = 200
    env = gym.make('CartPole-v1', render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    episode_reward_list = deque(maxlen=50)  # 创建了一个最大长度为 50 的双端队列, 当双端队列中的元素数量超过 50 个时，最早添加的元素会自动从队列的另一端删除
    episode_reward_history = []
    episode_reward_history_last50 = deque(maxlen=50)
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state[0], [1, state_size])
        total_reward = 0
        for time in range(200):  # 200 steps
            action = agent.act(state, policyIndicator)  # choose action from S using e-greedy
            next_state, reward, done, _, _ = env.step(action)  # take action and observed R, S'
            # done 表示当前状态是否为终止状态（terminal state）
            total_reward += reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("done")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size,0,policyIndicator)
        episode_reward_list.append(total_reward)
        episode_reward_avg = np.array(episode_reward_list).mean()
        episode_reward_history.append(episode_reward_avg)
        episode_reward_history_last50.append(episode_reward_avg)
        print("episode: {}/{}, score: {}, e: {:.2}, last 50 ep. avg. rew.: {:.2f}"
              .format(e, EPISODES, total_reward, agent.epsilon, episode_reward_avg))
    agent.save("q_weights.h5")
    return episode_reward_history


def Sarsa_epsilon(policy):
    if policy == "softmax":
        policyIndicator = 1
    else:
        policyIndicator = 0

    EPISODES = 200
    env = gym.make('CartPole-v1', render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    episode_reward_list = deque(maxlen=50)  # 创建了一个最大长度为 50 的双端队列, 当双端队列中的元素数量超过 50 个时，最早添加的元素会自动从队列的另一端删除
    episode_reward_history = []
    episode_reward_history_last50 = deque(maxlen=50)
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state[0], [1, state_size])
        total_reward = 0
        for time in range(200):  # 200 steps
            action = agent.act(state, policyIndicator)  # choose action from S using e-greedy
            next_state, reward, done, _, _ = env.step(action)  # take action and observed R, S'
            # done 表示当前状态是否为终止状态（terminal state）
            total_reward += reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("done")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size,1,policyIndicator)
        episode_reward_list.append(total_reward)
        episode_reward_avg = np.array(episode_reward_list).mean()
        episode_reward_history.append(episode_reward_avg)
        episode_reward_history_last50.append(episode_reward_avg)
        print("episode: {}/{}, score: {}, e: {:.2}, last 50 ep. avg. rew.: {:.2f}"
              .format(e, EPISODES, total_reward, agent.epsilon, episode_reward_avg))
    agent.save("Sarsa_weights.h5")
    return episode_reward_history


def test_model_CarPole(policy, name):
    if policy == "softmax":
        policyIndicator = 1
    else:
        policyIndicator = 0
    print(policyIndicator)

    EPISODES = 1

    env = gym.make('CartPole-v1', render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.load(name)
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state[0], [1, state_size])
        total_reward = 0
        for time in range(1000):
            action = agent.exploit(state)
            next_state, reward, done, _, _ = env.step(action)  # take action and observed R, S'
            total_reward += reward
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state
            if done:
                print("done")
                return total_reward
        return total_reward

training = 1
all_rewards_histories = []
if training == 1:
    # Q_epsilon = q_learning_epsilon("epsilon")
    # all_rewards_histories.append(Q_epsilon)

    # Q_softmax = q_learning_soft()
    # all_rewards_histories.append(Q_softmax)

    SARSA_epsilon = Sarsa_epsilon("epsilon")
    all_rewards_histories.append(SARSA_epsilon)

    # SARSA_softmax = Sarsa_epsilon("softmax")
    # all_rewards_histories.append(SARSA_softmax)

    plt.figure()
    for i, rewards_history in enumerate(all_rewards_histories):
        plt.plot(range(len(rewards_history)), rewards_history, label=f'Policy {i+1}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.show()

else:
    total_reward = test_model_CarPole("epsilon", "q_softmax_weights.h5")
    print(total_reward)
    time.sleep(1)