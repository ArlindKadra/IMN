from torch.utils.data import Dataset


class TabularDataset(Dataset):

    def __init__(self, numerical_features, categorical_features, labels):

        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.labels = labels
        print(labels.size)
    def __len__(self):
        return self.labels.size

    def __getitem__(self, idx):

        return self.numerical_features[idx], self.categorical_features[idx], self.labels[idx]
