import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(),
    target_transform=Lambda(lambda y : torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

'''
ToTensor()
PIL image 나 NumPy ndarray를 floatTensor로 변환하고, 이미지 픽셀 크기 값을 [0., 1.] 범위로 조정
Lambda 변형
정수를 원-핫으로 부호화된 텐서로 바꾸는 함수 정의
크기 10짜리 zero tensor 생성 후 scatter_ 호출해 주어진 label y에 해당하는 인덱스에 value=1을 할당
'''
target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))