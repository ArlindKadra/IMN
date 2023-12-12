import torch
from torch.utils.data import Dataset

class ContextDataset(Dataset):
    def __init__(self, X, Y, num_positive_examples=16, num_negative_examples=16):
        self.X = X
        self.Y = Y
        self.num_positive_examples = num_positive_examples
        self.num_negative_examples = num_negative_examples
        self.num_examples_per_point = num_positive_examples + num_negative_examples

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]

        positive_indices = torch.where(self.Y == y)[0]
        negative_indices = torch.where(self.Y != y)[0]

        # Sample positive and negative examples
        positive_samples = self.X[torch.randperm(len(positive_indices))[:self.num_positive_examples]]
        negative_samples = self.X[torch.randperm(len(negative_indices))[:self.num_negative_examples]]

        # Combine positive and negative examples
        sampled_examples = torch.cat((positive_samples, negative_samples), dim=0)

        # Create the final example with shape (11, 20)
        final_example = torch.cat((x.unsqueeze(0), sampled_examples), dim=0)

        return final_example, y