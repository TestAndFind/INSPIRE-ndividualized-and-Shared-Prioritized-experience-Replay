import time

import numpy as np
np.random.seed(1)
import torch
import torch.nn.functional as F

class PER_Memory(object):  
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    # epsilon = 0.01  # small amount to avoid zero priority
    # alpha = 0.6  # [0~1] convert the importance of TD error to priority
    # beta = 0.4  # importance-sampling, from initial value increasing to 1
    # beta_increment_per_sampling = 0.001
    # abs_err_upper = 1.  # clipped abs error

    def __init__(self, args, td_error, mask):
        self.alpha = args.selected_alpha
        self.res = torch.zeros_like(td_error)
        self.B, self.T, self.N =  td_error.shape
        self.mask = mask
        epsilon = mask * args.selected_epsilon
        td_error_epi = torch.abs(td_error) + epsilon
        td_error_epi_alpha = td_error_epi ** self.alpha
        self.prob = (td_error_epi_alpha / td_error_epi_alpha.sum())
        
        # beta
        self.max_step = args.t_max
        self.beta_start = args.beta_start
        self.beta_end = args.beta_end

    def sample(self, n):
        sampled_pos = torch.multinomial(self.prob, n, replacement=True)        
        index = sampled_pos
        pos_2 = index % self.N
        index = index // self.N
        pos_1 = index % self.T
        index = index // self.T
        pos_0 = index % self.B
        for i in range(n):
            self.res[pos_0[i],pos_1[i],pos_2[i]] += 1

        return self.res
    
    def sample_weight(self, n, step):
        sampled_pos = torch.multinomial(self.prob.reshape(-1), n, replacement=True) 
        N = self.B * self.T * self.N
        beta = (self.beta_end - self.beta_start)*step/self.max_step + self.beta_start
        weight = torch.pow(1 / (self.prob * N + 1e-8), beta) * self.mask 
        norm_weight = weight/ weight.max()

        index = sampled_pos
        pos_2 = index % self.N
        index = index // self.N
        pos_1 = index % self.T
        index = index // self.T
        pos_0 = index % self.B
        for i in range(n):
            self.res[pos_0[i],pos_1[i],pos_2[i]] += norm_weight[pos_0[i],pos_1[i],pos_2[i]]
        return self.res


#更新，新增函数形式
def sample_weight(selected_num, t_env, args, td_error, mask):
    #函数版本，计算损失权重
    #输入：选中tderror数目selected_num;轮次t_env;参数args;待计算权重的td_error;tderror的掩膜mask
    #输出：权重矩阵res

    #提取必要变量
    alpha = args.selected_alpha #tderror的指数参数
    B, T, N = td_error.shape #记录tderror1-3维度格式
    epsilon = mask * args.selected_epsilon #参数
    max_step = args.t_max #最大时间步
    beta_start = args.beta_start #beta参数的开始
    beta_end = args.beta_end#beta参数的最终值

    #调整tderror
    td_error_epi = torch.abs(td_error) + epsilon
    td_error_epi_alpha = td_error_epi ** alpha
    #计算概率
    prob = (td_error_epi_alpha / td_error_epi_alpha.sum())
    #构建权重
    sampled_pos = torch.multinomial(prob.reshape(-1), selected_num, replacement=True)#按照概率选取tderror
    total_num = B * T * N #计算tderror个数
    beta = (beta_end - beta_start) * t_env / max_step + beta_start #计算beta参数
    weight = torch.pow(1 / (prob * total_num + 1e-8), beta) * mask #计算权重
    norm_weight = weight/ weight.max() #归一化

    #更新，填写权重，新方法
    #建立mask，mask的格式同tderror，元素意义为tderror中这个位置被选中了多少次
    one_hot_vectors = F.one_hot(sampled_pos, total_num) #将坐标配置为一组one-hot向量
    mask = one_hot_vectors.sum(dim=0).reshape(td_error.shape) #one-hot向量求和，之后重组为tderror的格式
    #通过掩膜来计算res
    res = mask * norm_weight

    #填写权重,旧方法
    #res = torch.zeros_like(td_error)#存权重，格式和tderror一样
    # start = time.time()
    # index = sampled_pos
    # pos_2 = index % N
    # index = index // N
    # pos_1 = index % T
    # index = index // T
    # pos_0 = index % B
    # for i in range(selected_num):
    #     res[pos_0[i],pos_1[i],pos_2[i]] += norm_weight[pos_0[i],pos_1[i],pos_2[i]]
    # print("旧方法耗时：", time.time() - start)
    #
    # for i in range(res.shape[0]):
    #     for j in range(res.shape[1]):
    #         for k in range(res.shape[2]):
    #             if new_res[i][j][k].sum() != res[i][j][k].sum():
    #                 print("error in",i,j,k)
    #                 print("new_res值为",new_res[i][j][k],"res值为",res[i][j][k])


    return res


