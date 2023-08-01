# Model-free value function prediction method 2:

# Temporal Difference: model-free, learns from incomplete episodes by bootstrapping
# TD error: the difference between the estimated return of the new observation/new step and estimated value function

# step = 1: MC
# step = 0: DP

# SARSA
# On-policy TD Control: Single policy for sampling and update
# one-step
# n-steps: same as one-step in general, replace r+gamma*Q with r+expec(gamma*q(n))
# 通过e-greedy采样得到下一个state和reward,再通过e-greedy取新state的action

# Q-Learning
# Off-policy TD Control: Two policies, one for sampling, one for update
# 通过e-greedy采样,再通过greedy取让下一状态价值函数最大化的action, 更激进
