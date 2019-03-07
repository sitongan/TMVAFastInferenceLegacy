import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
       return x+y

a = torch.tensor([[1.,2.],[3.,4.]])
b = torch.tensor([[5.,6.],[7.,8.]])
model = Model()
print(model(a,b))

torch.onnx.export( model, (a,b), "SimpleAdd.onnx", verbose=True)
