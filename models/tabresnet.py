import torch
import torch.nn as nn


class TabResNet(nn.Module):
    def __init__(
            self,
            nr_features: int = 32,
            nr_classes: int = 10,
            nr_blocks: int = 2,
            hidden_size: int = 64,
            dropout_rate: float = 0.25,
            **kwargs,
    ):
        super(TabResNet, self).__init__()
        self.nr_blocks = nr_blocks
        self.hidden_size = hidden_size
        self.blocks = nn.ModuleList()
        self.batch_norm = nn.BatchNorm1d(self.hidden_size)
        self.act_func = torch.nn.GELU()
        self.nr_features = nr_features
        self.nr_classes = nr_classes
        self.input_layer = nn.Linear(nr_features, hidden_size)
        self.input_dropout = nn.Dropout(dropout_rate)

        for i in range(nr_blocks):
            self.blocks.append(self.make_residual_block(hidden_size, hidden_size, dropout_rate=dropout_rate))

        self.output_layer = nn.Linear(hidden_size, nr_classes)

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, self.BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def forward(self, x):

        x = x.view(-1, self.nr_features)

        x = self.input_layer(x)
        x = self.batch_norm(x)
        x = self.act_func(x)
        x = self.input_dropout(x)

        for i in range(self.nr_blocks):
            x = self.blocks[i](x)

        x = self.output_layer(x)

        return x

    def make_residual_block(self, in_features, output_features, dropout_rate=0.25):

        return self.BasicBlock(in_features, output_features, dropout_rate=dropout_rate)

    class BasicBlock(nn.Module):

        def __init__(self, in_features, output_features, dropout_rate=0.25):
            super(TabResNet.BasicBlock, self).__init__()
            self.dropout_rate = dropout_rate
            self.linear1 = nn.Linear(in_features, output_features)
            self.bn1 = nn.BatchNorm1d(output_features)
            self.gelu = nn.GELU()
            self.linear2 = nn.Linear(output_features, output_features)
            self.bn2 = nn.BatchNorm1d(output_features)
            self.hidden_state_dropout = nn.Dropout(self.dropout_rate)
            self.residual_dropout = nn.Dropout(self.dropout_rate)

        def forward(self, x):
            residual = x
            residual = self.residual_dropout(residual)

            out = self.linear1(x)
            out = self.bn1(out)
            out = self.gelu(out)
            out = self.hidden_state_dropout(out)

            out = self.linear2(out)
            out = self.bn2(out)

            out += residual
            out = self.gelu(out)

            return out
