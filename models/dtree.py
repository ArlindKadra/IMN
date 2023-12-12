import torch
import torch.nn as nn

class DTree(nn.Module):
    def __init__(
            self,
            nr_features: int = 32,
            nr_classes: int = 10,
            tree_depth: int = 2,
            **kwargs,
    ):
        super(DTree, self).__init__()
        self.extra_args = kwargs
        self.act_func = torch.nn.ReLU()
        self.sigmoid_func = torch.nn.Sigmoid()
        self.softmax_func = torch.nn.Softmax(dim=1)
        self.nr_features = nr_features
        self.nr_classes = nr_classes
        self.tree_depth = tree_depth
        self.nr_nodes = (2 ** self.tree_depth) - 1
        self.nr_leaf_nodes = 2 ** self.tree_depth

        self.feature_importances = torch.nn.Parameter(torch.rand(self.nr_nodes * nr_features, 1, requires_grad=True))
        self.feature_splits = torch.nn.Parameter(torch.rand(self.nr_nodes * nr_features, 1, requires_grad=True))
        self.leaf_node_classes = torch.nn.Parameter(torch.rand(self.nr_leaf_nodes * nr_classes, 1, requires_grad=True))

    def forward(self, x, return_weights: bool = False, discretize: bool = False, return_tree: bool = False):

        x = x.view(-1, self.nr_features)
        #feature_importances = x @ self.feature_importances
        #feature_splits = x @  self.feature_splits
        #leaf_node_classes = x @ self.leaf_node_classes

        feature_importances = torch.split(self.feature_importances, self.nr_features, 0)
        feature_splits = self.sigmoid_func(self.feature_splits)
        feature_splits = torch.split(feature_splits, self.nr_features, dim=0)
        leaf_node_classes = torch.split(self.leaf_node_classes, self.nr_classes, dim=0)

        leaf_node_contribs = []
        for leaf_node_index in range(0, self.nr_leaf_nodes):

            coefficient = torch.ones((x.size(0), leaf_node_classes[leaf_node_index].size(0)), device=x.device)
            for depth_index in range(1, self.tree_depth + 1):
                index_of_node = int((2 ** (depth_index - 1) * (2 ** self.tree_depth + leaf_node_index) - 2 ** self.tree_depth) / 2 ** self.tree_depth)
                p = int((leaf_node_index / 2 ** (self.tree_depth - depth_index)) % 2)
                softmaxed_feature_importances = self.act_func(feature_importances[index_of_node])

                if not discretize:
                    b = x @ softmaxed_feature_importances
                    c = softmaxed_feature_importances.transpose(1, 0) @ feature_splits[index_of_node]
                    node_sd = torch.sigmoid(torch.sub(b, c.squeeze()))
                else:
                    # get the max index of each row of the softmaxed feature importances
                    max_indices = torch.argmax(softmaxed_feature_importances, dim=1).unsqueeze(1)
                    feature_value = x[torch.arange(softmaxed_feature_importances.size(0)), max_indices.squeeze()]
                    feature_split = feature_splits[index_of_node][torch.arange(softmaxed_feature_importances.size(0)), max_indices.squeeze()]
                    node_sd = torch.sub(feature_value, feature_split)
                    node_sd = torch.sigmoid(node_sd)
                    # threshold
                    node_sd = torch.where(node_sd > 0.5, torch.ones_like(node_sd), torch.zeros_like(node_sd))

                coefficient *= node_sd * (1 - p) + (1 - node_sd) * p

            leaf_node_contribs.append(leaf_node_classes[leaf_node_index] * coefficient)

        output = sum(leaf_node_contribs)

        softmaxed_feature_importances = []
        for i in range(0, self.nr_nodes):
            softmaxed_feature_importances.append(feature_importances[i])

        softmaxed_feature_importances = sum(softmaxed_feature_importances)

        # replicate the feature importances for every example
        #feature_importances = feature_importances.repeat(x.size(0), 0)

        if return_weights:
            if return_tree:
                return output, softmaxed_feature_importances, (feature_importances, feature_splits, leaf_node_classes)
            else:
                return output, softmaxed_feature_importances

        else:
            return output
