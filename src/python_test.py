import torch
import torch.nn.functional as F
import torch.sparse as sparse

# Define sample matrix a and probability matrix b
a = torch.randint(1, 10, size=(2, 5, 4))
b = torch.rand((2, 5, 4))  # Probability matrix

# Number of samples
num_samples = 10

# Reshape probability matrix into 1D tensor
reshaped_b = b.view(-1)  # Flatten b along dimension 0

# Sample indices with replacement
indices = torch.multinomial(reshaped_b, num_samples, replacement=True)

# Combine indices into 3D index tensor
# coord_tensor = torch.stack([pos_0,pos_1,pos_2], dim=1)

num_elements = a.numel()
values = torch.ones(len(indices)).long()
sparse_one_hot = sparse.SparseTensor(size=(num_elements, num_elements), indices=indices.unsqueeze(1), values=values)

# Sum sparse one-hot vectors
sparse_mask = sparse_one_hot.sum(dim=0)

# Convert to dense tensor
mask = sparse_mask.to_dense().reshape(a.shape)

print(mask.shape)
#我有一个2*5*4的矩阵a，以及分别记录10个随机第一维坐标的数组pos_0，10个第二维坐标的数组pos_1，10个第三维坐标的数组pos_2
print("end")
