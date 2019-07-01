import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
       return x+y+z

size = 10
aaa = torch.rand(size)
bbb = torch.rand(size)
ccc = torch.rand(size)

a = torch.tensor([[1.,2.],[3.,4.]])
b = torch.tensor([[5.,6.],[7.,8.]])
c = torch.tensor([[9.,10.],[11.,12.]])
model = Model()



torch.onnx.export( model, (aaa,bbb,ccc), "SimpleAdd.onnx", verbose=True)
