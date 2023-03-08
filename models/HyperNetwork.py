import torch
import torch.nn as nn


class HyperNet(nn.Module):
    def __init__(
            self,
            nr_features: int = 32,
            nr_classes: int = 10,
            nr_blocks: int = 0,
            hidden_size: int = 64,
            dropout_rate: float = 0.2,
            **kwargs,
    ):
        super(HyperNet, self).__init__()
        self.nr_blocks = nr_blocks
        self.hidden_size = hidden_size
        self.blocks = nn.ModuleList()
        self.dropout_rate = dropout_rate
        self.batch_norm = nn.BatchNorm1d(self.hidden_size)
        self.act_func = torch.nn.SELU()
        self.nr_features = nr_features
        self.nr_classes = nr_classes

        self.input_layer = nn.Linear(nr_features, hidden_size)

        for i in range(nr_blocks):
            self.blocks.append(self.make_residual_block(hidden_size, hidden_size))

        self.output_layer = nn.Linear(hidden_size, nr_features * nr_classes)

    def forward(self, x, return_weights: bool = False):

        x = x.view(-1, self.nr_features)
        input = x

        x = self.input_layer(x)
        x = self.batch_norm(x)
        x = self.act_func(x)
        for i in range(self.nr_blocks):
            residual = x
            x = self.blocks[i](x)
            x = x + residual
            x = self.act_func(x)

        #x = self.act_func(x)
        w = self.output_layer(x)
        w = w.view(-1, self.nr_features, self.nr_classes)
        x = torch.einsum("ij,ijk->ik", input, w)

        if return_weights:
            return x, w
        else:
            return x

    def make_residual_block(self, in_features, output_features):

        return nn.Sequential(

            nn.Linear(in_features, output_features),
            nn.BatchNorm1d(output_features),
            self.act_func,
            nn.Linear(output_features, output_features),
            nn.BatchNorm1d(output_features),
        )