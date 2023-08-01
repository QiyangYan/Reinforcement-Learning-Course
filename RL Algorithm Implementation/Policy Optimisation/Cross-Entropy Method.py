# CEM for black-box function optimization
'''
This is used for derivative-free value functions for value function approximator that requries optimisation
简单来说,就是先随机初始化一个概率分布,从中sampling,然后从新的sample里选 J(theta) 最大的一部分sample,
然后算他们平均值和standard deviation,并用这个来得到新的概率分布,再对新的distribution sampling,反复这个过程
'''

import numpy as np

def cem(f, th_mean, batch_size, n_iter, elite_frac, initial_std=1.0):
    """
      f: a function mapping from vector -> scalar
      th_mean: initial mean over input distribution
      batch_size: number of samples of theta to evaluate per batch
      n_iter: number of batches
      elite_frac: each batch, select this fraction of the top-performing samples
      initial_std: initial standard deviation over parameter vectors
      """
    n_elite = int(np.round(batch_size * elite_frac))  # number of samples that will be used for update from each batch
    th_std = np.ones_like(th_mean) * initial_std

    for _ in range(n_iter):
        # STEP1: Sample theta from an initialised distribution
        ths = np.array([th_mean + dth for dth in
                        th_std[None, :] * np.random.randn(batch_size,th_mean.size)])
        '''
        np.random.randn(): generates an array of shape (batch_size, th_mean.size) 
        containing random samples from a standard normal distribution.
        
        th_std and th_mean are updated in every loop, use updated parameter to update the sample distribution for sampling
        得到来自具有均值为 th_mean 和标准差为 th_std 的正态分布的随机样本
        th_std[None, :]: reshapes the th_std array to have shape (1, th_mean.size), 
        one-dimensional array to two-dimensional
        + th_mean: 得到来自具有均值为 th_mean 的随机样本
        
        np.array(): 转换为 NumPy 数组    

        dth for dth in ARRAY: 是列表推导式, generates a new list from (iterable) array. 在每次迭代时从array中取出一个元素，并将其作为新的元素加入到 list 中。
        expression for item in iterable: 其中 expression 是一个用于生成新元素的表达式，item 是从可迭代对象 iterable 中逐个取出的元素。
        列表推导式在这里的效果和不加是一样的,但是有以下好处
        1. 用列表推导式的方式更紧凑
        2. 方便对每个样本进行单独操作
        3. 可读性较好
        '''

        # STEP2: Evaluate J(theta)
        ys = np.array([f(th) for th in ths])
        '''
        apply function f(th) to every element th obtained from sampled iterable array ths, and store them as a new list
        '''

        # STEP3: Get part of the list that has the highest value function
        elite_inds = ys.argsort()[::-1][:n_elite]
        elite_ths = ths[elite_inds] # C{}
        '''
        argsort() 函数返回的索引值表示按照从小到大的顺序排列原始数组中的元素，并且这些索引值构成一个新的数组。
        ys.argsort()[::-1] 这是将排序后的索引值进行逆序（从大到小）排列
        [:n_elite] 得到前 n_elite 个样本的索引
        ths[elite_inds] 用得到的索引从ths里挑选sample
        '''

        # STEP 4: calculate new parameters for update
        th_mean = elite_ths.mean(axis=0)
        th_std = elite_ths.std(axis=0)

        yield {'ys': ys, 'theta_mean': th_mean, 'y_mean': ys.mean()}