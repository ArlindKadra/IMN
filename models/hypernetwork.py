import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from models.exu import LogLinear


class HyperNet(nn.Module):
    def __init__(
        self,
        nr_features: int = 32,
        nr_classes: int = 10,
        nr_blocks: int = 0,
        hidden_size: int = 64,
        dropout_rate: float = 0.25,
        unit_type: str = "basic",
        **kwargs,
    ):
        super(HyperNet, self).__init__()
        self.nr_blocks = nr_blocks
        self.hidden_size = hidden_size
        self.blocks = nn.ModuleList()
        self.batch_norm = nn.BatchNorm1d(self.hidden_size)
        self.act_func = torch.nn.GELU()
        self.nr_features = nr_features
        self.nr_classes = nr_classes
        self.input_layer = nn.Linear(nr_features, hidden_size)
        self.dropout_rate = dropout_rate
        self.output_drop = nn.Dropout(self.dropout_rate)

        for _ in range(nr_blocks):
            self.blocks.append(self.make_residual_block(hidden_size, hidden_size, dropout_rate=self.dropout_rate))

        if unit_type != "basic":
            self.output_layer = LogLinear(hidden_size, (nr_features + 1) * nr_classes)
        else:
            self.output_layer = nn.Linear(hidden_size, (nr_features + 1) * nr_classes, bias=False)

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, self.BasicBlock) and m.bn2.weight is not None:
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x, return_weights: bool = False, simple_weights: bool = False):

        x = x.view(-1, self.nr_features)
        input = x

        x = self.input_layer(x)
        x = self.batch_norm(x)
        x = self.act_func(x)
        x = self.output_drop(x)

        for i in range(self.nr_blocks):
            x = self.blocks[i](x)

        w = self.output_layer(x)
        #w = self.output_drop(w)

        input = torch.cat((input, torch.ones(input.shape[0], 1).to(x.device)), dim=1)
        w = w.view(-1, (self.nr_features + 1), self.nr_classes)
        x = torch.einsum("ij,ijk->ik", input, w)
        #w = w.squeeze(2)
        repeated_input = torch.stack([input for _ in range(self.nr_classes)], dim=2)
        # if test mode
        if not self.training and not simple_weights:

            w_without_bias = w[:, :-1, :]
            feature_vectors = []
            for i in range(w_without_bias.size(1)):

                feature_i = w_without_bias[:, i, :]
                # create figure
                feature_i = torch.flatten(feature_i.detach()).cpu().numpy()
                feature_vectors.append(feature_i)

            fig = plt.figure()
            plt.violinplot(feature_vectors)
            #plt.hist(torch.flatten(feature_i.detach()).cpu().numpy())
            #plt.ylabel('Frequency')
            plt.xlabel('Feature')
            plt.ylabel('Feature Weights')
            plt.title("Feature Weight Distributions")
            plt.savefig("feature_weights.pdf", bbox_inches='tight')
            plt.close(fig)

            w = repeated_input[:, :-1, :] * w[:, :-1, :]

        if return_weights:
            return x,  w
        else:
            return x

    def make_residual_block(
        self,
        in_features: int,
        output_features: int,
        dropout_rate: float = 0.25,
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

        return self.BasicBlock(in_features, output_features, dropout_rate)

    class BasicBlock(nn.Module):

        def __init__(
            self,
            in_features: int,
            output_features: int,
            dropout_rate: float,
        ):
            super(HyperNet.BasicBlock, self).__init__()
            self.dropout_rate = dropout_rate
            self.hidden_state_dropout = nn.Dropout(self.dropout_rate)
            self.residual_dropout = nn.Dropout(self.dropout_rate)
            self.linear1 = nn.Linear(in_features, output_features)
            self.bn1 = nn.BatchNorm1d(output_features)
            self.gelu = nn.GELU()
            self.linear2 = nn.Linear(output_features, output_features)
            self.bn2 = nn.BatchNorm1d(output_features)

        def forward(self, x):
            residual = x
            residual = self.residual_dropout(residual)

            out = self.linear1(x)
            out = self.bn1(out)
            out = self.gelu(out)
            out = self.hidden_state_dropout(out)
            out = self.linear2(out)
            out = self.bn2(out)
            out = self.gelu(out)

            out += residual

            return out