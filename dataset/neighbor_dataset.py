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

        # find the unique classes
        unique_classes = torch.unique(self.Y)

        # remove the y label from the unique classes
        unique_classes = unique_classes[unique_classes != y]

        # pick a random class from the remaining classes
        random_class_index = torch.randint(0, len(unique_classes), (1,))[0]
        random_class = unique_classes[random_class_index]

        # Find indices of points with a different label
        different_label_indices = torch.where(self.Y == random_class)[0]

        # Calculate distances between x and all points with a different label
        distances = torch.cdist(x.view(-1, x.size(0)), self.X[different_label_indices])
        distances = torch.squeeze(distances)
        # Find the 10 closest points
        closest_indices = torch.argsort(distances)[:5]
        rand_closest_index = torch.randint(0, len(closest_indices), (1,))[0]
        rand_closest_index = closest_indices[rand_closest_index]
        closest_index = rand_closest_index
        # Find the index of the closest point
        #closest_index = different_label_indices[torch.argmin(distances)]

        # Get the closest point and its label
        closest_point = self.X[closest_index]
        closest_label = self.Y[closest_index]

        return x, y, closest_point, closest_label
