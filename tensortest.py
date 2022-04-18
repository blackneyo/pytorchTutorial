import torch
import numpy as np


# # 텐서(tensor) 초기화
# data = [[1,2], [3,4]]
# # 데이터 자료형은 자동 유추
# x_data = torch.tensor(data)
#
# # NumPy 배열로 텐서 생성
# np_array = np.array(data)
# x_np = torch.from_numpy(np_array)
#
# # x_data의 속성 유지 (인자로 주어진 shape, datatype)
# x_ones = torch.ones_like(x_data)
# print(f"Ones Tensor: \n {x_ones} \n")
#
# #x_data의 속성을 덮음(overide)
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
# # shape는 텐서의 차원을 나타내는 튜플 / 아래 함수들은 출력 텐서의 차원을 결정
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
# # 텐서의 속성
# # 텐서의 속성은 텐서의 shape, datatype 및 어느 장치에 저장되는지 나타냄
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
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

'''
텐서의 연산
transposing, indexing, slicing, 수학 계산, 선형 대수, random sampling 등 100가지 이상의 텐서 연산 존재
각 연산들은 GPU에서 실행은 가능하지만 텐서는 CPU에 생성
.to 메소드를 이용해 GPU로 이동 가능
'''
# NumPy 식의 표준 indexing & slicing
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
# print(tensor)
'''
First row: tensor([1., 1., 1., 1.])
First column: tensor([1., 1., 1., 1.])
Last column: tensor([1., 1., 1., 1.])
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
'''
# # 텐서 3개 연결 1차원배열 / torch.cat을 이용하여 일련의 텐서 연결
# t1 = torch.cat([tensor, tensor, tensor], dim=1)
# print(t1)
# '''
# tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
#         [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
#         [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
#         [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
# '''
#
# # 산출 연산
# # 두 텐서 간의 행렬 곱(matrix multiplication) 계산 / y1, y2, y3는 모두 같은 값을 가짐
# y1 = tensor @ tensor.T
# y2 = tensor.matmul(tensor.T)
# y3 = torch.rand_like(tensor)
#
# torch.matmul(tensor, tensor.T, out=y3)
#
# #요소별 곱 (element-wise product) 을 계산 / z1, z2, z3 는 모두 같은 값을 가짐
# z1 = tensor * tensor
# z2 = tensor.mul(tensor)
# z3 = torch.rand_like(tensor)
# torch.mul(tensor, tensor, out=z3)
#
# # 단일-요소텐서 = 텐서 모든 값을 하나로 집계(aggregate) 요소가 하나인 텐서의 경우 item() 을 사용하여 python 숫자 값으로 변환할 수 있음
# agg = tensor.sum()
# agg_item = agg.item()
# print(agg_item, type(agg_item))
# '''
# 12.0 <class 'float'>
# '''

# 바꿔치기(in-place) 연산
print(tensor, '\n')
tensor.add_(5)
print(tensor)
'''
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]]) 

tensor([[6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.]])
'''

# NumPy 변환(Bridge)
# 텐서를 NumPy 배열로 변환

t = torch.ones(5)
print(f't: {t}')
n = t.numpy()
print(f'n: {n}')
'''
t: tensor([1., 1., 1., 1., 1.])
n: [1. 1. 1. 1. 1.]
'''
# 텐서의 변경 사항이 NumPy 배열에 반영
t.add_(1)
print(f't:{t}')
print(f'n:{n}')
'''
t:tensor([2., 2., 2., 2., 2.])
n:[2. 2. 2. 2. 2.]
'''

# NumPy 배열을 텐서로 변환
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f't: {t}')
print(f'n: {n}')
'''
t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
n: [2. 2. 2. 2. 2.]
'''