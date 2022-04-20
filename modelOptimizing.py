import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

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

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stuck = nn.Sequential(
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

model = NeuralNetwork()

'''
하이퍼 파라미터(Hyperparameter)
epoch - 데이터 셋 반복 횟수
batch_size - 매개변수 갱신 전 신경망을 통해 전파된 데이터 샘플 수
learning rate - 각 배치/에포치에서 모델의 매개변수를 조절하는 비율. 값이 작을수록 학습 속도가 느려지고 값이 크면 학습 중 예측할 수 없는 동작이 발생할 수 있음
'''
learning_rate = 1e-3
batch_size = 64
epochs = 5

'''
최적화 단계(Optimization Loop)
하나의 epoch은 두 부분으로 구성
1. 학습단계 (train loop) - 학습용 데이터셋을 반복하고 최적의 매개변수로 수렴
2. 검증/테스트 단계(validation/test loop) - 모델 성능이 개선되고 있는지를 확인하기 위해 테스트 데이터셋을 반복

손실 함수(loss function)
획득한 결과와 실제 값 사이의 틀린 정도를 측정하여 학습 중에 이 값을 최소화
주어진 데이터 샘플을 입력으로 계산한 예측과 정답(label)을 비교하여 손실(loss)을 계산
회귀문제(regression task) = nn.MSELoss(평균 제곱 오차(Mean Square Error))
분류(classification) = nn.NLLLoss(Negative Log Likelihood)
nn.CrossEntropyLoss = nn.Logsoftmax + nn.NLLLoss
'''

loss_fn = nn.CrossEntropyLoss()

'''
학습 단계 (loop)에서 최적화는 3단계로 이루어짐
1. optimizer.zero_grad 를 호출해 모델 매개변수의 변화도 재설정
   변화도는 더해지기(add up) 때문에 중복 계산을 막기 위해 반복 시 명시적으로 0 설정
2. loss.backwards()를 호출하여 예측 손실(prediction loss)을 역전파
3. 변화도 계산 뒤에는 optimizer.step()을 호출하여 역전파 단계에서 수집된 변화도로 매개변수 조정
'''

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d/{size:>5d}}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss()