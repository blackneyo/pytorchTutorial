# pytorchTutorial

튜토리얼

#Log

Shape of X [N, C, H, W] torch.Size([64, 1, 28, 28])

Shape of y:torch.Size([64]) torch.int64

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
# Epoch 1
-----------------------------
loss: 2.305903  [    0/60000]
loss: 2.289810  [ 6400/60000]
loss: 2.272108  [12800/60000]
loss: 2.264099  [19200/60000]
loss: 2.239260  [25600/60000]
loss: 2.207342  [32000/60000]
loss: 2.218525  [38400/60000]
loss: 2.177626  [44800/60000]
loss: 2.185154  [51200/60000]
loss: 2.140248  [57600/60000]
Test Error: 
 Accuracy: 42.8%, Avg loss: 2.135803 

# Epoch 2
-----------------------------
loss: 2.158987  [    0/60000]
loss: 2.139385  [ 6400/60000]
loss: 2.076434  [12800/60000]
loss: 2.093523  [19200/60000]
loss: 2.031779  [25600/60000]
loss: 1.973597  [32000/60000]
loss: 2.002798  [38400/60000]
loss: 1.914823  [44800/60000]
loss: 1.932151  [51200/60000]
loss: 1.854946  [57600/60000]
Test Error: 
 Accuracy: 48.4%, Avg loss: 1.845709 

# Epoch 3
-----------------------------
loss: 1.896034  [    0/60000]
loss: 1.852123  [ 6400/60000]
loss: 1.729718  [12800/60000]
loss: 1.778026  [19200/60000]
loss: 1.669706  [25600/60000]
loss: 1.623966  [32000/60000]
loss: 1.648945  [38400/60000]
loss: 1.548085  [44800/60000]
loss: 1.583122  [51200/60000]
loss: 1.479816  [57600/60000]
Test Error: 
 Accuracy: 59.4%, Avg loss: 1.492090 

# Epoch 4
-----------------------------
loss: 1.574209  [    0/60000]
loss: 1.530888  [ 6400/60000]
loss: 1.378252  [12800/60000]
loss: 1.454435  [19200/60000]
loss: 1.347130  [25600/60000]
loss: 1.336403  [32000/60000]
loss: 1.348709  [38400/60000]
loss: 1.276925  [44800/60000]
loss: 1.312455  [51200/60000]
loss: 1.216501  [57600/60000]
Test Error: 
 Accuracy: 62.9%, Avg loss: 1.240571 

# Epoch 5
-----------------------------
loss: 1.326974  [    0/60000]
loss: 1.304474  [ 6400/60000]
loss: 1.136788  [12800/60000]
loss: 1.244342  [19200/60000]
loss: 1.128435  [25600/60000]
loss: 1.145185  [32000/60000]
loss: 1.162772  [38400/60000]
loss: 1.107138  [44800/60000]
loss: 1.141827  [51200/60000]
loss: 1.062648  [57600/60000]
Test Error: 
 Accuracy: 64.4%, Avg loss: 1.081776 


Done!

Saved PyTorch Model State to model.pth

Predicted: "Ankle boot", Actual: "Ankle boot"

Process finished with exit code 0
