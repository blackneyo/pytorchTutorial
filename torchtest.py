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
    print(f"Shape of y:{y.shape} {y.dtype}")
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
# 모델 학습하려면 손실 함수(loss function) 옵티마이저(optimizer) 필요
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 예측 오류 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100  == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-----------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
'''
Epoch 1
-----------------------------
loss: 2.311845  [    0/60000]
loss: 2.293704  [ 6400/60000]
loss: 2.271029  [12800/60000]
loss: 2.269351  [19200/60000]
loss: 2.243505  [25600/60000]
loss: 2.225886  [32000/60000]
loss: 2.228145  [38400/60000]
loss: 2.202638  [44800/60000]
loss: 2.194429  [51200/60000]
loss: 2.149966  [57600/60000]
Test Error:
 Accuracy: 44.0%, Avg loss: 2.157614

Epoch 2
-----------------------------
loss: 2.172496  [    0/60000]
loss: 2.155881  [ 6400/60000]
loss: 2.094960  [12800/60000]
loss: 2.111393  [19200/60000]
loss: 2.047864  [25600/60000]
loss: 2.006040  [32000/60000]
loss: 2.019603  [38400/60000]
loss: 1.950688  [44800/60000]
loss: 1.956498  [51200/60000]
loss: 1.859134  [57600/60000]
Test Error:
 Accuracy: 60.9%, Avg loss: 1.874680

Epoch 3
-----------------------------
loss: 1.915570  [    0/60000]
loss: 1.876528  [ 6400/60000]
loss: 1.756100  [12800/60000]
loss: 1.791973  [19200/60000]
loss: 1.671867  [25600/60000]
loss: 1.638646  [32000/60000]
loss: 1.642803  [38400/60000]
loss: 1.555162  [44800/60000]
loss: 1.579213  [51200/60000]
loss: 1.455989  [57600/60000]
Test Error:
 Accuracy: 61.2%, Avg loss: 1.491412

Epoch 4
-----------------------------
loss: 1.564103  [    0/60000]
loss: 1.525939  [ 6400/60000]
loss: 1.374379  [12800/60000]
loss: 1.441859  [19200/60000]
loss: 1.319334  [25600/60000]
loss: 1.330357  [32000/60000]
loss: 1.330574  [38400/60000]
loss: 1.264904  [44800/60000]
loss: 1.297647  [51200/60000]
loss: 1.189940  [57600/60000]
Test Error:
 Accuracy: 63.1%, Avg loss: 1.226397

Epoch 5
-----------------------------
loss: 1.307380  [    0/60000]
loss: 1.288485  [ 6400/60000]
loss: 1.119117  [12800/60000]
loss: 1.225099  [19200/60000]
loss: 1.098061  [25600/60000]
loss: 1.135727  [32000/60000]
loss: 1.147548  [38400/60000]
loss: 1.091704  [44800/60000]
loss: 1.127065  [51200/60000]
loss: 1.039101  [57600/60000]
Test Error:
 Accuracy: 64.9%, Avg loss: 1.067337

'''

# 모델 저장하기
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

'''
Saved PyTorch Model State to model.pth
'''

# 모델 불러오기
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))


# 모델 사용해서 예측
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
