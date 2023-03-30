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
        self.act_func = torch.nn.ReLU()
        self.nr_features = nr_features
        self.nr_classes = nr_classes

        self.input_layer = nn.Linear(nr_features, hidden_size)

        for i in range(nr_blocks):
            self.blocks.append(self.make_residual_block(hidden_size, hidden_size))

        self.output_layer = nn.Linear(hidden_size, nr_features * nr_classes)

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, self.BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def forward(self, x, return_weights: bool = False):

        x = x.view(-1, self.nr_features)
        input = x

        x = self.input_layer(x)
        x = self.batch_norm(x)
        x = self.act_func(x)

        for i in range(self.nr_blocks):
            x = self.blocks[i](x)

        w = self.output_layer(x)
        w = w.view(-1, self.nr_features, self.nr_classes)
        x = torch.einsum("ij,ijk->ik", input, w)

        if return_weights:
            return x, w
        else:
            return x

    def make_residual_block(self, in_features, output_features):

        return self.BasicBlock(in_features, output_features)

    class BasicBlock(nn.Module):

        def __init__(self, in_features, output_features):
            super(HyperNet.BasicBlock, self).__init__()
            self.linear1 = nn.Linear(in_features, output_features)
            self.bn1 = nn.BatchNorm1d(output_features)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(output_features, output_features)
            self.bn2 = nn.BatchNorm1d(output_features)

        def forward(self, x):
            residual = x

            out = self.linear1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.linear2(out)
            out = self.bn2(out)

            out += residual
            out = self.relu(out)

            return out
