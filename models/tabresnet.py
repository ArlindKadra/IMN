import torch
import torch.nn as nn

from utils import BasicBlock, make_residual_block


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

        for _ in range(nr_blocks):
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
