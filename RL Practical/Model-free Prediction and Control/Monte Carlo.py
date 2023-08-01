# Model-free value function prediction method 1:

# ------------------ MC Prediction ---------
# MC Prediction: 通过采样多段完整Complete的有终点terminal的轨迹,计算当前policy下每个状态的return,
# 加和每一条轨迹每一个state的收获,总收获(empirical mean return)的平均值就是“价值函数 value function”的近似解
# Keywards: Sampling, Terminal, Empirical mean return


# 其他注意事项:
# 1. 完整序列中重复出现同样状态,那么该状态的收获该如何计算？有两种解决方法:
#   首次访问(first visit): 第一种是仅把状态序列中第一次出现该状态时的收获值纳入到收获平均值的计算中；
#   每次访问(every visit): 另一种是针对一个状态序列中每次出现的该状态，都计算对应的收获值并纳入到收获平均值的计算中。
#   第二种方法比第一种的计算量要大一些，但是在完整的经历样本序列少的场景下会比第一种方法适用。

# 2. Incremental mean
#   储存每一个收获最后再取均值太浪费储存空间.迭代计算收获均值Incremental MC update，即每次保存上一轮迭代得到的收获均值与次数，
#   当计算得到当前轮的收获时，即可用上一轮结果计算当前轮收获均值和次数.
#   同时这样也有利于计算non-stationary problems,因为这样是随机应变
#   alpha(learning rate): 因为很难确定迭代的次数N(St),所以用alpha来替换该项,表示每次被新的学习影响多少.

#  MC Control


#  ------------------ Next: TD ------------------
# 虽然蒙特卡罗法很灵活，不需要环境的状态转化概率模型，但是它需要所有的采样序列都是经历"完整的状态序列"。
# 如果我们没有完整的状态序列，那么就无法使用蒙特卡罗法求解了。
# 下面我们就来讨论可以不使用完整状态序列求解强化学习问题的方法：时序差分(Temporal-Difference, TD)。只需要下一步/n步的observation

# Advantages of MC over DP: page 14, Lecture 3
# Difference between MC and TD: page 18, Lecture 3
# 笔记链接: https://zhuanlan.zhihu.com/p/319028715