import torch
import torch.nn as nn


class FactorizedHyperNet(nn.Module):
    def __init__(
        self,
        nr_features: int = 32,
        nr_classes: int = 10,
        nr_blocks: int = 0,
        hidden_size: int = 64,
        factor_size: int = 4,
        **kwargs,
    ):
        super(FactorizedHyperNet, self).__init__()
        self.nr_blocks = nr_blocks
        self.hidden_size = hidden_size
        self.blocks = nn.ModuleList()
        self.batch_norm = nn.BatchNorm1d(self.hidden_size)
        self.act_func = torch.nn.GELU()
        self.nr_features = nr_features
        self.nr_classes = nr_classes
        self.input_layer = nn.Linear(nr_features, hidden_size)
        self.second_head = nn.Linear(hidden_size, nr_classes)
        self.factor_size = factor_size

        for _ in range(nr_blocks):
            self.blocks.append(self.make_residual_block(hidden_size, hidden_size))

        self.output_layer = nn.Linear(hidden_size, (nr_features + 1) * nr_classes)
        self.extra_complexity = nn.Linear(hidden_size, nr_features * self.factor_size * nr_classes)

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, self.BasicBlock) and m.bn2.weight is not None:
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x, return_weights: bool = False):

        x = x.view(-1, self.nr_features)
        input = x

        x = self.input_layer(x)
        x = self.batch_norm(x)
        x = self.act_func(x)

        for i in range(self.nr_blocks):
            x = self.blocks[i](x)

        w = self.output_layer(x)

        input_matrix = input.view(-1, self.nr_features, 1)
        input_matrix = torch.matmul(input_matrix, input_matrix.transpose(1, 2))
        factorized_weights = self.extra_complexity(x)
        factorized_weights = factorized_weights.view(-1, self.nr_features, self.factor_size, self.nr_classes)
        class_fact_weights = torch.split(factorized_weights, 1, dim=3)


        class_info = []
        class_additions = []
        for i in range(len(class_fact_weights)):
            specific_class_weights = class_fact_weights[i]
            specific_class_weights = torch.squeeze(specific_class_weights, dim=3)
            specific_class_weights = torch.matmul(specific_class_weights, specific_class_weights.transpose(1, 2))
            factorized_info = torch.matmul(specific_class_weights, input_matrix)
            batch_size = factorized_info.shape[0]
            factorized_result = torch.stack([torch.triu(factorized_info[i, :, :], diagonal=1) for _ in range(batch_size)])
            class_additions.append(torch.sum(factorized_result, dim=1))
            factorized_result = torch.sum(factorized_result, dim=[1, 2])
            class_info.append(factorized_result)

        factorized_info = torch.stack(class_info, 1)
        class_additions = torch.stack(class_additions, 2)
        input = torch.cat((input, torch.ones(input.shape[0], 1).to(x.device)), dim=1)
        w = w.view(-1, (self.nr_features + 1), self.nr_classes)
        x = torch.einsum("ij,ijk->ik", input, w)
        x = x + factorized_info

        #weight_additions = input[:, :-1] * w[:, :-1, 0]
        #class_additions = torch.squeeze(class_additions, dim=2)
        #weight_additions = weight_additions + class_additions

        if return_weights:
            return x, w
        else:
            return x

    def make_residual_block(
        self,
        in_features: int,
        output_features: int,
    ):
        """Creates a residual block.

        Args:
            in_features: int
                Number of input features to the first
                layer of the residual block.
            output_features: Number of output features
                for the last layer of the residual block.

        Returns:
            BasicBlock
                A residual block.
        """

        return self.BasicBlock(in_features, output_features)

    class BasicBlock(nn.Module):

        def __init__(
            self,
            in_features: int,
            output_features: int,
        ):
            super(FactorizedHyperNet.BasicBlock, self).__init__()
            self.linear1 = nn.Linear(in_features, output_features)
            self.bn1 = nn.BatchNorm1d(output_features)
            self.gelu = nn.GELU()
            self.linear2 = nn.Linear(output_features, output_features)
            self.bn2 = nn.BatchNorm1d(output_features)

        def forward(self, x):
            residual = x

            out = self.linear1(x)
            out = self.bn1(out)
            out = self.gelu(out)
            out = self.linear2(out)
            out = self.bn2(out)

            out += residual
            out = self.gelu(out)

            return out