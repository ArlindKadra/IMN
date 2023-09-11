import torch
import torch.nn as nn

class DTHyperNet(nn.Module):
    def __init__(
            self,
            nr_features: int = 32,
            nr_classes: int = 10,
            nr_blocks: int = 0,
            hidden_size: int = 64,
            tree_depth: int = 2,
            **kwargs,
    ):
        super(DTHyperNet, self).__init__()
        self.nr_blocks = nr_blocks
        self.hidden_size = hidden_size
        self.blocks = nn.ModuleList()
        self.batch_norm = nn.BatchNorm1d(self.hidden_size)
        self.act_func = torch.nn.GELU()
        self.sigmoid_func = torch.nn.Sigmoid()
        self.nr_features = nr_features
        self.nr_classes = nr_classes
        self.input_layer = nn.Linear(nr_features, hidden_size)
        self.tree_depth = tree_depth
        self.nr_nodes = (2 ** self.tree_depth) - 1
        self.nr_leaf_nodes = 2 ** self.tree_depth
        for i in range(nr_blocks):
            self.blocks.append(self.make_residual_block(hidden_size, hidden_size))

        self.feature_importances = nn.Linear(hidden_size, self.nr_nodes * nr_features)
        self.feature_splits = nn.Linear(hidden_size, self.nr_nodes * nr_features)
        self.leaf_node_classes = nn.Linear(hidden_size, self.nr_leaf_nodes * nr_classes)

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, self.BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def forward(self, x, return_weights: bool = False, discretize: bool = False):

        x = x.view(-1, self.nr_features)
        input = x

        x = self.input_layer(x)
        x = self.batch_norm(x)
        x = self.act_func(x)

        for i in range(self.nr_blocks):
            x = self.blocks[i](x)

        feature_importances = self.feature_importances(x)
        feature_splits = self.feature_splits(x)
        leaf_node_classes = self.leaf_node_classes(x)
        feature_importances = torch.split(feature_importances, self.nr_features, dim=1)
        feature_splits = torch.split(feature_splits, self.nr_features, dim=1)
        leaf_node_classes = torch.split(leaf_node_classes, self.nr_classes, dim=1)

        leaf_node_contribs = []
        for leaf_node_index in range(0, self.nr_leaf_nodes):

            coefficient = torch.ones(leaf_node_classes[leaf_node_index].size(), device=x.device)
            for depth_index in range(1, self.tree_depth + 1):
                index_of_node = int((2 ** (depth_index - 1) * (2 ** self.tree_depth + leaf_node_index) - 2 ** self.tree_depth) / 2 ** self.tree_depth)
                p = int((leaf_node_index / 2 ** (self.tree_depth - depth_index)) % 2)
                softmaxed_feature_importances = torch.softmax(feature_importances[index_of_node], dim=1)
                if not discretize:
                    node_sd = torch.sigmoid(torch.sub(torch.sum(softmaxed_feature_importances * input, dim=1), torch.sum(softmaxed_feature_importances * feature_splits[index_of_node], dim=1)))
                else:
                    # get the max index of each row of the softmaxed feature importances
                    max_indices = torch.argmax(softmaxed_feature_importances, dim=1).unsqueeze(1)
                    feature_value = input[torch.arange(softmaxed_feature_importances.size(0)), max_indices.squeeze()]
                    feature_split = feature_splits[index_of_node][torch.arange(softmaxed_feature_importances.size(0)), max_indices.squeeze()]
                    node_sd = torch.sub(feature_value, feature_split)
                    node_sd = torch.sigmoid(node_sd)
                    # threshold
                    node_sd = torch.where(node_sd > 0.5, torch.ones_like(node_sd), torch.zeros_like(node_sd))

                coefficient *= node_sd[:, None] * (1 - p) + (1 - node_sd)[:, None] * p
            leaf_node_contribs.append(leaf_node_classes[leaf_node_index] * coefficient)

        output = sum(leaf_node_contribs)

        softmaxed_feature_importances = []
        for i in range(0, self.nr_nodes):
            softmaxed_feature_importances.append(torch.softmax(feature_importances[i], dim=1))

        feature_importances = sum(softmaxed_feature_importances)

        if return_weights:
            return output, feature_importances
        else:
            return output

    def make_residual_block(self, in_features, output_features):

        return self.BasicBlock(in_features, output_features)

    class BasicBlock(nn.Module):

        def __init__(self, in_features, output_features):
            super(DTHyperNet.BasicBlock, self).__init__()
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
