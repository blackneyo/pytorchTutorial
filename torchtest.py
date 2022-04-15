# import torch
# x = torch.rand(5, 3)
# print(x)

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 공개 데이터셋에서 학습 데이터를 사용
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# 공개 데이터셋에서 테스트 데이터 사용
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

'''
dataset을 dataloader 인자로 전달
자동화된 batch, sampling, shuffle, multiprocess data loading 지원
배치 사이즈 64
즉, 데이터로더(dataloader) 객체의 각 요소는 64개의 특징(feature)과 정답(label)을 묶음(batch)로 반환
'''
batch_size = 64

# 데이터 로더를 생성
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W] {X.shape}")
    print(f"Shape of y:{y.shape}{y.dtype}")
    break

'''
Shape of X [N, C, H, W] torch.Size([64, 1, 28, 28])
Shape of y:torch.Size([64])torch.int64
'''

# 학습에 사용할 CPU나 GPU 장치
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# 모델 정의
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

'''
Using cpu device
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
'''

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

'''
Saved PyTorch Model State to model.pth
'''