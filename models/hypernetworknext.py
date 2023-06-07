import torch
import torch.nn as nn


class HyperNet(nn.Module):
    def __init__(
            self,
            nr_features: int = 32,
            nr_classes: int = 10,
            nr_blocks: int = 2,
            hidden_size: int = 64,
            dropout_rate: float = 0.2,
            cardinality: int = 4,
    ):
        super(HyperNet, self).__init__()
        self.nr_levels = nr_blocks
        self.hidden_size = hidden_size

        self.dropout_rate = dropout_rate
        self.cardinality = cardinality
        self.act_func = torch.nn.SELU()
        self.batch_norm = nn.BatchNorm1d(self.hidden_size)

        self.input_layer = nn.Linear(nr_features, hidden_size)
        for level_index in range(self.nr_levels):
            module_list = nn.ModuleList()
            for _ in range(cardinality):
                module_list.append(self.make_residual_block(self.hidden_size, self.hidden_size))
            setattr(self, f'level_{level_index}', module_list)

        self.output_layer = nn.Linear(hidden_size, (nr_features + 1) * nr_classes)
        self.nr_features = nr_features
        self.nr_classes = nr_classes

    def forward(self, x, return_weights: bool = False):

        x = x.view(-1, self.nr_features)
        input = x

        x = self.input_layer(x)
        x = self.batch_norm(x)
        x = self.act_func(x)

        for i in range(self.nr_levels):
            residual = x.clone().detach()
            blocks = getattr(self, f'level_{i}')
            for j in range(0, self.cardinality):
                x += blocks[j](residual)
            x = x + residual
            x = self.act_func(x)

        w = self.output_layer(x)
        # add column of ones to the input variable to account for the intercept
        input = torch.cat((input, torch.ones(input.shape[0], 1).to(x.device)), dim=1)
        w = w.view(-1, self.nr_features + 1, self.nr_classes)
        x = torch.einsum("ij,ijk->ik", input, w)

        if return_weights:
            return x, w
        else:
            return x

    def make_residual_block(
        self,
        in_features: int,
        output_features: int,
    ) -> nn.Sequential:
        """Creates a residual block.

        Args:
            in_features: int
                Number of input features to the first
                layer of the residual block.
            output_features: Number of output features
                for the last layer of the residual block.

        Returns:
            nn.Sequential
                A residual block.
        """
        lower_embedding = int(output_features / self.cardinality)
        return nn.Sequential(
            nn.Linear(in_features, lower_embedding),
            nn.BatchNorm1d(int(lower_embedding)),
            self.act_func,
            nn.Linear(lower_embedding, lower_embedding),
            nn.BatchNorm1d(lower_embedding),
            self.act_func,
            nn.Linear(lower_embedding, output_features),
            nn.BatchNorm1d(output_features),
        )
