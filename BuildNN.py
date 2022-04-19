import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
'''
신경망 모델을 nn.Module 의 하위클래스로 정의하고, __init__ 에서 신경망 계층들을 초기화
nn.module 을 상속받은 모든 클래스는 forward 메소드 입력 데이터에 대한 연산들 구현
'''
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
# print(model)
'''
NeuralNetwork의 인스턴스 생성
->device로 이동 후 구조(structure)출력
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
'''
모델 사용 하기 위해 데이터 전달
모델에 입력을 호출하면 각 분류에 대한 raw 예측값이 있는 10-차원 텐서가 반환
raw 예측값을 nn.Softmax 모듈의 인스턴스에 통과시켜 예측값을 얻음
'''
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

'''
Usiong cpu device
Predicted class: tensor([8])
'''

'''
모델 계층(Layer)
FashionMNIST 모델의 계층]
28x28 크기의 이미지 3개로 구성된 미니배치를 가져와 신경망을 통과
'''
input_image = torch.rand(3, 28, 28)
print(input_image.size())
'''
torch.Size([3, 28, 28])
'''
'''
nn.Flatten 계층을 초기화
28x28 2D 이미지를 784픽셀 값을 갖는 연속된 배열로 변환
'''
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())
'''
torch.Size([3, 784])
'''
'''
nn.linear
저장된 가중치(weight)와 bias를 사용하여 입력에 선형 변환(linear transform)적용
'''
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())
'''
torch.Size([3, 20])
'''
'''
nn.ReLU
'''
print(f"Before ReLU: {hidden1}:\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
'''
nn.Sequential
순서를 갖는 모듈의 컨테이너
sequential container를 사용하여 아래의 seq_modules와 같은 신경망을 빠르게 만듬
'''
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)

'''
nn.sortmax모듈에 전달될 raw value인 logits를 반환
logits는 모델의 각 class 에 대한 예측 확률을 나타내도록 [0, 1] 범위로 비례하여 조정
dim 매개변수는 값의 합이 1이 되는 차원을 나타냄
'''
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

'''
모델 매개변수
nn.Module을 상속하면 모델 객체 내부의 모든 필드들이 자동으로 tracking되며
모델의 parameters() 및 named_parameters() 메소드로 모든 매개변수에 접근 가능
'''
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")