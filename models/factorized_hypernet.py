import torch
import torch.nn as nn
from models.exu import LogLinear

class FactorizedHyperNet(nn.Module):
    def __init__(
        self,
        nr_features: int = 32,
        nr_classes: int = 10,
        nr_blocks: int = 0,
        hidden_size: int = 64,
        factor_size: int = 4,
        dropout_rate: float = 0.2,
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
        self.dropout_rate = dropout_rate

        for _ in range(nr_blocks):
            self.blocks.append(self.make_residual_block(hidden_size, hidden_size))

        self.output_layer = LogLinear(hidden_size, (nr_features + 1) * nr_classes)
        self.extra_complexity = LogLinear(hidden_size, nr_features * self.factor_size * nr_classes)
        self.interaction_dropout = nn.Dropout(self.dropout_rate)
        self.main_weights_dropout = nn.Dropout(self.dropout_rate)

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, self.BasicBlock) and m.bn2.weight is not None:
                nn.init.constant_(m.bn2.weight, 0)

    def reinitialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x, return_weights: bool = False):

        x = x.view(-1, self.nr_features)
        input = x

        x = self.input_layer(x)
        x = self.batch_norm(x)
        x = self.act_func(x)

        for i in range(self.nr_blocks):
            x = self.blocks[i](x)

        w = self.output_layer(x)
        w = self.main_weights_dropout(w)
        # b x n x 1
        input_matrix = input.view(-1, self.nr_features, 1)

        # b x n x n
        input_matrix = torch.einsum("nij,njk->nik", input_matrix, input_matrix.transpose(1, 2))

        factorized_weights = self.extra_complexity(x)
        factorized_weights = self.interaction_dropout(factorized_weights)

        # b x c x n x k
        factorized_weights = factorized_weights.view(self.nr_classes, -1,  self.nr_features, self.factor_size)

        # c x b x n x n
        factorized_weights = torch.einsum("iljk, ilkm->iljm", factorized_weights, factorized_weights.transpose(2, 3))

        #factorized_info = torch.einsum("cbnj, bnj->cbnj", factorized_weights, input_matrix)
        factorized_info = factorized_weights * input_matrix
        mask = torch.triu(torch.ones(self.nr_features, self.nr_features), diagonal=1).to(x.device)
        #factorized_info = torch.einsum("cbnj, nj -> cbnj", factorized_info * mask)
        factorized_info = factorized_info * mask
        class_additions_part1 = torch.sum(factorized_info, dim=3)
        class_additions_part2 = torch.sum(factorized_info, dim=2)
        class_additions = class_additions_part1 + class_additions_part2
        factorized_info = torch.sum(factorized_info, dim=(2, 3))

        factorized_info = factorized_info.permute(1, 0)
        class_additions = class_additions.permute(1, 2, 0)

        input = torch.cat((input, torch.ones(input.shape[0], 1).to(x.device)), dim=1)
        w = w.view(-1, (self.nr_features + 1), self.nr_classes)
        x = torch.einsum("ij,ijk->ik", input, w)
        x = x + factorized_info
        #input = input.view(-1, self.nr_features + 1, 1)
        repeated_input = torch.stack([input for _ in range(self.nr_classes)], dim=2)
        weight_additions = repeated_input[:, :-1] * w[:, :-1, :]
        weight_additions = weight_additions + 0.5 * class_additions

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