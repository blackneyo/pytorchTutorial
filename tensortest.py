import torch
import numpy as np


# # 텐서(tensor) 초기화
# data = [[1,2], [3,4]]
# x_data = torch.tensor(data)
#
# np_array = np.array(data)
# x_np = torch.from_numpy(np_array)
#
# # x_data의 속성 유지
# x_ones = torch.ones_like(x_data)
# print(f"Ones Tensor: \n {x_ones} \n")
#
# #x_data의 속성을 덮음
# x_rand = torch.rand_like(x_data, dtype=torch.float)
# print(f"Random Tensor: \n {x_rand} \n")
# '''
# Ones Tensor:
#  tensor([[1, 1],
#         [1, 1]])
#
# Random Tensor:
#  tensor([[0.2029, 0.8984],
#         [0.9665, 0.4603]])
# '''
# # 무작위(random) 또는 상수(constant) 값 사용
# shape = (2, 3)
# rand_tensor = torch.rand(shape)
# ones_tensor = torch.ones(shape)
# zeros_tensor = torch.zeros(shape)
#
# print(f"Random Tensor: \n {rand_tensor} \n")
# print(f"Ones Tensor: \n {ones_tensor} \n")
# print(f"Zeros Tensor: \n {zeros_tensor}")
#
# '''
# Random Tensor:
#  tensor([[0.1440, 0.1614, 0.3452],
#         [0.6744, 0.0012, 0.0966]])
#
# Ones Tensor:
#  tensor([[1., 1., 1.],
#         [1., 1., 1.]])
#
# Zeros Tensor:
#  tensor([[0., 0., 0.],
#         [0., 0., 0.]])
# '''
# tensor = torch.ones(4, 4)
# print(f"First row: {tensor[0]}")
# print(f"First column: {tensor[:,0]}")
# print(f"Last column: {tensor[..., -1]}")
# tensor[:,1] = 0
# print(tensor)
# '''
# First row: tensor([1., 1., 1., 1.])
# First column: tensor([1., 1., 1., 1.])
# Last column: tensor([1., 1., 1., 1.])
# tensor([[1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.]])
#
# '''
#
# if torch.cuda.is_available():
#     tensor = tensor.to("cuda")
#
# NumPy 식의 표준 인덱싱과 슬라이싱
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)
'''
First row: tensor([1., 1., 1., 1.])
First column: tensor([1., 1., 1., 1.])
Last column: tensor([1., 1., 1., 1.])
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
'''
# 텐서 3개 연결 1차원배열
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
'''
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
'''
