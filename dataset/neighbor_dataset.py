import torch
from torch.utils.data import Dataset

class ContextDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.num_samples = X.shape[0]

        examples_closest = []
        unique_labels = torch.unique(self.Y)
        for example_index in range(self.num_samples):

            x = self.X[example_index]
            y = self.Y[example_index]

            # remove y from unique_labels
            unique_labels = unique_labels[unique_labels != y]

            # Find indices of points with a different label
            different_label_indices = torch.where(self.Y != y)[0]

            # Calculate distances between x and all points with a different label
            distances = torch.cdist(x.view(-1, x.size(0)), self.X[different_label_indices])
            distances = torch.squeeze(distances)
            # Find the 10 closest points
            closest_indices = torch.argsort(distances)[:10]
            examples_closest.append(closest_indices)

        self.examples_closest = torch.stack(examples_closest, dim=0)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index].item()

        rand_closest_index = torch.randint(0, len(self.examples_closest[index]), (1,))[0]
        rand_closest_index = self.examples_closest[index][rand_closest_index]
        # Find the index of the closest point
        #closest_index = different_label_indices[torch.argmin(distances)]

        # Get the closest point and its label
        closest_point = self.X[rand_closest_index]
        closest_label = self.Y[rand_closest_index]

        return x, y, closest_point, closest_label
