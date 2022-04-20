'''
2022.04.20 이동한
'''
import torch
'''
매개변수 (모델 가중치)는 주어진 매개변수에 대한 변화도 (gradient) 에 따라 조정
pytorch에는 torch.autograd라는 자동 미분 엔진이 내장
모든 계산 그래프에 대한 변화도의 자동 계산 지원
'''
# 아래는 입력 x, 매개변수 w, b 일부 손실 함수가 있는 가장 간단한 단일계층 신경망

x = torch.ones(5) # input tensor
y = torch.zeros(3) # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

'''
requires_grad 의 값은 텐서를 생성할 때 설정하거나,
x.requires_grad_(True)메소드 사용하여 나중에 설정 가능
'''

print(f"Gradient fuction for z = {z.grad_fn}")
print(f"Gradient fuction for loss = {loss.grad_fn}")

'''
Gradient fuction for z = <AddBackward0 object at 0x0000025FF38BEDC0>
Gradient fuction for loss = <BinaryCrossEntropyWithLogitsBackward0 object at 0x0000025FF38BEDC0>
'''

'''
변화도 계산
매개변수 가중치의 최적화를 위해 매개변수에 대한 손실함수의 도함수 계산
loss.backward() 호출
'''
loss.backward()
print(w.grad)
print(b.grad)
'''
tensor([[0.3121, 0.1480, 0.0696],
        [0.3121, 0.1480, 0.0696],
        [0.3121, 0.1480, 0.0696],
        [0.3121, 0.1480, 0.0696],
        [0.3121, 0.1480, 0.0696]])
tensor([0.3121, 0.1480, 0.0696])
'''
'''
변화도 추적 멈추기
requires_grad=True인 모든 텐서들은 연산 기록을 추적하고 변화도 계산 지원
torch.no_grad() 블록으로  둘러싸서 연산 추적 멈춤
'''
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)
'''
True
False
'''

z =torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

'''
False
'''

'''
변화도 추적을 멈춰야 하는 이유
1. 신경망의 일부 매개변수를 고정된 매개변수로 표시. 이는 사전 학습된 신경망을 미세조정 할 때 매우 일반적인 시나리오
2. 변화도를 추적하지 않는 텐서의 연산이 더 효율적 / 연산 속도 향상
'''

inp = torch.eye(5, requires_grad=True)
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")
