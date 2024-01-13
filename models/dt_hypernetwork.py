import torch
import torch.nn as nn
from models.exu import LogLinear

class DTHyperNet(nn.Module):
    def __init__(
            self,
            nr_features: int = 32,
            nr_classes: int = 10,
            nr_blocks: int = 2,
            hidden_size: int = 64,
            tree_depth: int = 2,
            **kwargs,
    ):
        super(DTHyperNet, self).__init__()
        self.sigmoid_func = torch.nn.Sigmoid()
        #self.set_transformer = SetTransformer(dim_input=nr_features, num_outputs=1, dim_output=nr_features, num_heads=4, ln=False)
        self.nr_blocks = nr_blocks
        self.hidden_size = hidden_size
        self.blocks = nn.ModuleList()
        self.batch_norm = nn.BatchNorm1d(self.hidden_size)
        self.act_func = torch.nn.ReLU()
        self.leaky_relu = torch.nn.LeakyReLU()
        self.nr_features = nr_features
        self.nr_classes = nr_classes
        self.input_layer = nn.Linear(nr_features, hidden_size)
        self.tree_depth = tree_depth
        self.nr_nodes = (2 ** self.tree_depth) - 1
        self.nr_leaf_nodes = 2 ** self.tree_depth
        for i in range(nr_blocks):
            self.blocks.append(self.make_residual_block(hidden_size, hidden_size))

        self.feature_importances = LogLinear(hidden_size, self.nr_nodes * nr_features)
        self.feature_splits = LogLinear(hidden_size, self.nr_nodes * nr_features)
        self.leaf_node_classes = LogLinear(hidden_size, self.nr_leaf_nodes * nr_classes)

        #torch.nn.init.xavier_uniform_(self.feature_importances.weight)
        #torch.nn.init.xavier_uniform_(self.feature_splits.weight)
        #torch.nn.init.xavier_uniform_(self.leaf_node_classes.weight)

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, self.BasicBlock) and m.bn2.weight is not None:
                nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]



    def calculate_predictions(
        self,
        x,
        feature_importances,
        feature_splits,
        leaf_node_classes,
        discretize: bool = False,
        return_tree: bool = False,
    ):

        leaf_node_contribs = []

        """
        if discretize and return_tree:
            for feature_importance, feature_split in zip(feature_importances, feature_splits):
                softmaxed_feature_importance = self.act_func(feature_importance)
                max_indices = torch.argmax(softmaxed_feature_importance, dim=1).unsqueeze(1)
                #feature_value = x[torch.arange(softmaxed_feature_importance.size(0)), max_indices.squeeze()]
                #feature_split = feature_split[torch.arange(softmaxed_feature_importance.size(0)), max_indices.squeeze()]
                #tree_features.append(max_indices)
                #tree_splits.append(feature_split)

        """
        class_additions = [torch.zeros(feature_importances[0].size(), device=x.device)] * self.nr_classes

        for leaf_node_index in range(0, self.nr_leaf_nodes):
            feature_output_additions = torch.ones(feature_importances[0].size(), device=x.device)
            coefficient = torch.ones(leaf_node_classes[leaf_node_index].size(), device=x.device)
            for depth_index in range(1, self.tree_depth + 1):
                index_of_node = int((2 ** (depth_index - 1) * (
                            2 ** self.tree_depth + leaf_node_index) - 2 ** self.tree_depth) / 2 ** self.tree_depth)
                p = int((leaf_node_index / 2 ** (self.tree_depth - depth_index)) % 2)
                softmaxed_feature_importances = torch.softmax(feature_importances[index_of_node], dim=1)
                #softmaxed_feature_importances = self.act_func(feature_importances[index_of_node])

                if not discretize:
                    node_sd = torch.sigmoid(torch.sum(softmaxed_feature_importances * x, dim=1) - torch.sum(
                        softmaxed_feature_importances * feature_splits[index_of_node], dim=1))
                    feature_output_additions_node = torch.sigmoid(
                        softmaxed_feature_importances * x - softmaxed_feature_importances * feature_splits[index_of_node])
                else:
                    # get the max index of each row of the softmaxed feature importances
                    max_indices = torch.argmax(softmaxed_feature_importances, dim=1).unsqueeze(1)
                    feature_value = x[torch.arange(softmaxed_feature_importances.size(0)), max_indices.squeeze()]
                    feature_split = feature_splits[index_of_node][
                        torch.arange(softmaxed_feature_importances.size(0)), max_indices.squeeze()]
                    node_sd = torch.sub(feature_value, feature_split)
                    node_sd = torch.sigmoid(node_sd)
                    # threshold
                    node_sd = torch.where(node_sd > 0.5, torch.ones_like(node_sd), torch.zeros_like(node_sd))

                node_sd = node_sd.view(-1, 1)
                coefficient *= node_sd * (1 - p) + (1 - node_sd) * p

                feature_output_additions *= (1 - p) * feature_output_additions_node + p * feature_output_additions_node

            for class_additions_index in range(0, self.nr_classes):
                bla = leaf_node_classes[leaf_node_index][:, class_additions_index]
                bla = bla.view(-1, 1)
                class_additions[class_additions_index] += feature_output_additions * bla

            leaf_node_contribs.append(leaf_node_classes[leaf_node_index] * coefficient)

        output = sum(leaf_node_contribs)

        return output, class_additions

    def forward(
        self,
        x,
        return_weights: bool = False,
        discretize: bool = False,
        return_tree: bool = False,
    ):

        x = x.view(-1, self.nr_features)

        initial_input = x

        x = self.input_layer(x)
        x = self.batch_norm(x)
        x = self.act_func(x)

        for i in range(self.nr_blocks):
            x = self.blocks[i](x)

        feature_importances = self.feature_importances(x)
        feature_splits = self.feature_splits(x)
        #feature_splits = self.sigmoid_func(feature_splits)
        leaf_node_classes = self.leaf_node_classes(x)
        feature_importances = torch.split(feature_importances, self.nr_features, dim=1)
        feature_splits = torch.split(feature_splits, self.nr_features, dim=1)
        leaf_node_classes = torch.split(leaf_node_classes, self.nr_classes, dim=1)

        output, class_additions = self.calculate_predictions(
            initial_input,
            feature_importances,
            feature_splits,
            leaf_node_classes,
            discretize,
            return_tree,
        )

        softmaxed_feature_importances = []
        for i in range(0, self.nr_nodes):
            softmaxed_feature_importances.append(torch.softmax(feature_importances[i], dim=1))

        softmaxed_feature_importances = sum(softmaxed_feature_importances)

        feature_splits = torch.concat(feature_splits, dim=1)
        if self.training:
            class_additions = feature_splits

        if return_weights:
            if return_tree:
                return output, class_additions, (feature_importances, feature_splits, leaf_node_classes)
            else:
                return output, class_additions
        else:
            return output

    def make_residual_block(self, in_features, output_features):

        return self.BasicBlock(in_features, output_features)

    class BasicBlock(nn.Module):

        def __init__(self, in_features, output_features, dropout_rate=0.25):
            self.dropout_rate = dropout_rate
            super(DTHyperNet.BasicBlock, self).__init__()
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
            out = self.gelu(out)

            out += residual

            return out
