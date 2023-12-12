import torch
from torch.utils.data import Dataset

class ContextDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.num_samples = X.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index].item()

        # Find indices of points with a different label
        different_label_indices = torch.where(self.Y != y)[0]

        # Calculate distances between x and all points with a different label
        distances = torch.cdist(x.unsqueeze(0), self.X[different_label_indices])

        # Find the index of the closest point
        closest_index = different_label_indices[torch.argmin(distances)]

        # Get the closest point and its label
        closest_point = self.X[closest_index]
        closest_label = self.Y[closest_index]

        return x, y, closest_point, closest_label
