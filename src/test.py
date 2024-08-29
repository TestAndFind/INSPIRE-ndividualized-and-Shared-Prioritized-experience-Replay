import numpy as np
import torch as th

k = 4
m =3

a = th.ones(10, 4, 3) * 4
a[0][0][0] = 8
a[0][0][1] = 9
up = th.ones(10, 4, 3) * 6
down = th.ones(10, 4, 3) * 3

receive_list_1 = th.zeros(10,4,3) # 第三维存所有接受的tderror的和
for dim in range(3):  # 逐个智能体计算门掩膜，重组share数据
    # 修改share_list格式
    share_list_dim = a[:, :, dim]  # 截取分享清单第dim智能体分享的数据
    share_list_dim = share_list_dim.unsqueeze(2).repeat(1, 1, 3)  # 增加第三维，把agent[dim]的经验数据铺给每个agent
    share_list_dim[:, :, dim] = th.zeros_like(share_list_dim[:, :, dim])  # 删去分享给自己的经验
    # 制作掩膜
    receive_q_mask = (share_list_dim >= up) + ((share_list_dim <= down) * (share_list_dim !=0))
    receive_q_mask = receive_q_mask
    # 获取筛选后结果
    receive_list_dim = receive_q_mask * share_list_dim
    # 结果加入总记录
    receive_list_1[receive_q_mask] = receive_list_dim[receive_q_mask]
    # receive_list_1 = receive_list_1 + receive_list_dim
receive_list_1 = receive_list_1

receive_list = th.zeros(10,4,3)
for i in range(4):
    for j in range(4):
        dim = 0
        for agent in range(3):
            if agent == dim:
                receive_list[i][j][agent] = 0 #不接受自己发出的经验
            else:
                if a[i][j][dim] >= up[i][j][agent] or a[i][j][dim] <= down[i][j][agent]:
                    # 如果分享单子上记着的tderror在自己这边也在均值+-标准差之外，记录
                    receive_list[i][j][agent] = a[i][j][dim]#修改，改为加和
        dim = dim + 1

for i in range(receive_list.shape[0]):
    for j in range(receive_list.shape[1]):
        for k in range(receive_list.shape[2]):
            if receive_list_1[i][j][k] != receive_list[i][j][k]:
                print("error:",receive_list_1[i][j][k],receive_list[i][j][k])
print("end")

# a[0][0][0] = 2
# print(a,a.shape)
# # print(np.sum(a,axis=1), np.sum(a,axis=1).shape)
# mean = np.sum(a,axis=1)/a.shape[1]
# print(mean,mean.shape)
#
# print(np.std(a,axis=1))
#
# arr = np.zeros((k,m,2))
# print(arr)
# print(arr.shape[0])

# a = np.arange(12).reshape(3,4,1)
# print(a,a.shape)
# print('*&*************************************')
# print(a[:,:,0:1],a[:,:,0:1].shape)
# print(a[:,:,0:1],np.sum(a[:,:,0:1],axis=1))
# print('*&*************************************')
# print(a[:,:,1:2])
# print('*&*************************************')
# print(a[:,:,2:3])
# print('*&*************************************')