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
        self.second_head = nn.Linear(hidden_size, nr_classes)
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

    def forward(self, x, return_weights: bool = False):

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

        leaf_outputs = []
        for individual_leaf_classes in leaf_node_classes:
            leaf_outputs.append(individual_leaf_classes)
        #first_leaf_output = leaf_node_classes[0]
        #second_leaf_output = leaf_node_classes[1]
        #third_leaf_output = leaf_node_classes[2]
        #fourth_leaf_output = leaf_node_classes[3]

        feature_probabilities = []
        for node_feature_importances in feature_importances:
            feature_probabilities.append(torch.softmax(node_feature_importances, dim=1))
        #first_part_feature_importance = torch.softmax(feature_importances[0], dim=1)
        #second_part_feature_importance = torch.softmax(feature_importances[1], dim=1)
        #third_part_feature_importance = torch.softmax(feature_importances[2], dim=1)

        leaf_node_contribs = []
        for leaf_node_index in range(0, self.nr_leaf_nodes):
            leaf_node = leaf_outputs[leaf_node_index]
            coefficient = torch.ones(leaf_node.size(), device=x.device)
            for depth_index in range(0, self.tree_depth):
                index_of_node = int((2 ** (depth_index - 1) * (2 ** self.tree_depth + leaf_node_index) - 2 ** self.tree_depth) / 2 ** self.tree_depth)
                p = int((leaf_node_index / 2 ** (self.tree_depth - depth_index)) % 2)
                softmaxed_feature_importances = torch.softmax(feature_importances[index_of_node], dim=1)
                node_sd = torch.sigmoid(torch.sum(softmaxed_feature_importances * input, dim=1) - torch.sum(softmaxed_feature_importances * feature_splits[index_of_node], dim=1))
                coefficient *= node_sd[:, None] * (1 - p) + (1 - node_sd)[:, None] * p
            leaf_node_contribs.append(leaf_node * coefficient)

        #first_node_first_part = first_part_feature_importance * input
        #first_node_second_part = first_part_feature_importance * feature_splits[0]
        #second_node_first_part = second_part_feature_importance * input
        #second_node_second_part = second_part_feature_importance * feature_splits[1]
        #third_node_first_part = third_part_feature_importance * input
        #third_node_second_part = third_part_feature_importance * feature_splits[2]
        #first_node_first_part = torch.sum(first_node_first_part, dim=1)
        #first_node_second_part = torch.sum(first_node_second_part, dim=1)
        #second_node_first_part = torch.sum(second_node_first_part, dim=1)
        #second_node_second_part = torch.sum(second_node_second_part, dim=1)
        #third_node_first_part = torch.sum(third_node_first_part, dim=1)
        #third_node_second_part = torch.sum(third_node_second_part, dim=1)

        #first_node = self.sigmoid_func(first_node_first_part - first_node_second_part)
        #second_node = self.sigmoid_func(second_node_first_part - second_node_second_part)
        #third_node = self.sigmoid_func(third_node_first_part - third_node_second_part)

        #first_node = torch.amax(first_node, dim=1)
        #second_node = torch.amax(second_node, dim=1)
        #third_node = torch.amax(third_node, dim=1)

        #output = first_leaf_output * first_node[:, None] * second_node[:, None]
        #output += second_leaf_output * first_node[:, None] * (1 - second_node)[:, None]
        #output += third_leaf_output * (1 - first_node)[:, None] * third_node[:, None]
        #output += fourth_leaf_output * (1 - first_node)[:, None] * (1 - third_node)[:, None]
        #feature_importances = first_part_feature_importance + second_part_feature_importance + third_part_feature_importance

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
