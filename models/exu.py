import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class LogLinear(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LogLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = Parameter(torch.Tensor(out_dim, in_dim))
        self.bias = Parameter(torch.Tensor(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        self.weight = torch.nn.init.normal_(self.weight, mean=3, std=0.5)
        self.bias = torch.nn.init.normal_(self.bias, mean=3, std=0.5)

    def forward(self, inp):
        output = inp
        output = output.matmul(torch.exp(self.weight.t()))
        output = output - self.bias
        #output = output.matmul(self.weight.t())
        #output = F.leaky_relu(output)

        return output