import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

'''
root 는 학습/테스트 데이터가 저장되는 경로
train 은 학습/테스스용 데이터셋 여부 지정
download=True는 root에 데이터가 없는 경우 인터넷에서 다운로드
transform 과 target_transform은 특징(feature)과 정답(Label) 변형(transform) 지정
'''
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

'''
데이터셋 순회/시각화
'''

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows +1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
'''
figure_1 을 png 파일로 저장
'''

import os
import pandas as pd
from torchvision.io import read_image

'''
dataset 클래스는 반드시 3개의 함수 구현을 필요로 함
__init__ : dataset 객체가 생성될 때 한번만 실행됨
__len__ : dataset의 샘플 갯수 반환
__getitem__ : 주어진 인덱스 idx에 해당하는 샘플을 dataset에서 불러오고 반환
'''
class CustomImageDataSet(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # read_image를 사용하여 이미지를 텐서로 변환
        image = read_image(img_path)
        # self.img_labels의 csv 데이터로 부터 해당하는 label을 가져오고 transform 함수 호출한 뒤 텐서 이미지와 label을 python dictionary 형태로 반환
        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
        return image, label

'''
DataLoader로 학습용 데이터 준비하기
dataloader에 데이터셋을 불러온 뒤 필요에 따라 데이터셋을 반복
train_features 와 train_labels를 반환
Shuffle=True 로 지정 
'''
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Label batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
'''
Feature batch shape: torch.Size([64, 1, 28, 28])
Label batch shape: torch.Size([64])
Label: 3
'''
