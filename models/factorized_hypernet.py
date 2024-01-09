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
        for specific_class_weights in class_fact_weights:
            specific_class_weights = torch.squeeze(specific_class_weights, dim=3)
            """
            
            example_info = []
            for example_index, example in enumerate(specific_class_weights):
                current_info = torch.zeros((1), device=x.device)
                for k in range(example.size(1)):

                    first_part = torch.zeros((1), device=x.device)
                    for i in range(example.size(0)):
                        first_part += specific_class_weights[example_index, i, k] * input_matrix[example_index, i]
                    first_part = first_part ** 2
                    current_info += first_part
                    for i in range(example.size(0)):
                        current_info -= (specific_class_weights[example_index, i, k] ** 2) * (input_matrix[example_index, i] ** 2)
                example_info.append(current_info)
            example_info = torch.stack(example_info)
            example_info = torch.squeeze(example_info, dim=1)
            class_info.append(example_info)
            """
            specific_class_weights = torch.matmul(specific_class_weights, specific_class_weights.transpose(1, 2))
            factorized_info = torch.matmul(specific_class_weights, input_matrix)
            batch_size = factorized_info.shape[0]
            factorized_result = torch.stack([torch.triu(factorized_info[i, :, :], diagonal=1) for i in range(batch_size)])
            class_additions.append(torch.sum(factorized_result, dim=1) + torch.sum(factorized_result, dim=2))
            factorized_result = torch.sum(factorized_result, dim=[1, 2])
            class_info.append(factorized_result)

        class_info = torch.stack(class_info, 1)
        class_additions = torch.stack(class_additions, 2)
        input = torch.cat((input, torch.ones(input.shape[0], 1).to(x.device)), dim=1)
        w = w.view(-1, (self.nr_features + 1), self.nr_classes)
        x = torch.einsum("ij,ijk->ik", input, w)
        x = x + class_info
        #input = input.view(-1, self.nr_features + 1, 1)
        repeated_input = torch.stack([input for _ in range(self.nr_classes)], dim=2)
        weight_additions = repeated_input[:, :-1] * w[:, :-1, :]
        weight_additions = weight_additions + class_additions

        if return_weights:
            return x, weight_additions
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