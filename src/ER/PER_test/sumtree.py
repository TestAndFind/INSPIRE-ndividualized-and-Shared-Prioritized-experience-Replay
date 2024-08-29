import numpy as np


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        #初始化函数
        #输入：容量capacity
        self.capacity = capacity  # 计算容量
        self.tree = np.zeros(2 * capacity - 1) #使用树存储优先级和索引
        # [--------------树节点-------------][-------真正存储优先级值的叶节点-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # 存储经验
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        #函数：向求和树添加一个节点
        tree_idx = self.data_pointer + self.capacity - 1 #计算树上最近一个空的叶节点
        self.data[self.data_pointer] = data  # 更新经验池
        self.update(tree_idx, p)  # 更新树结构

        self.data_pointer += 1 #记录已经存了多少节点
        if self.data_pointer >= self.capacity:  # 超出限制时清空
            self.data_pointer = 0
    
    def update(self, tree_idx, p):
        #函数：从指定节点开始更新树结构
        #输入：树节点tree_idx，优先级值p
        change = p - self.tree[tree_idx] #计算变化量
        self.tree[tree_idx] = p #修改节点存储的优先级
        # 沿着树更新优先级值
        while tree_idx != 0:    #循环递推寻找父节点，更新其优先级值
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        #输入一个值v，找到从左往右求和到小于且最接近这个值的叶节点，返回
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # 计算左右子
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # 找到叶节点后返回
                leaf_idx = parent_idx
                break
            else:       # 递推寻找
                if v <= self.tree[cl_idx]: #左子大于和，进入左子
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx] #左子小于和，目标在右子，和减去左子后进入右子
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        #函数：返回优先级的总和
        return self.tree[0]  # 根据求和树性质，根节点的值就是总和

