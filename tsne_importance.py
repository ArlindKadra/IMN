import numpy as np
from sklearn.manifold import TSNE
from utils import get_dataset
import matplotlib.pyplot as plt
import openml

seed = 0
np.random.seed(seed)

dataset_id = 1590
test_split_size = 0.2
encoding_type = 'ordinal'
info = get_dataset(
    dataset_id,
    test_split_size=test_split_size,
    seed=seed,
    encoding_type=encoding_type,
)

dataset = openml.datasets.get_dataset(dataset_id, download_data=False)
dataset_name = dataset.name
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format='dataframe',
    target=dataset.default_target_attribute
)
X_train = info['X_train']
X_test = info['X_test']
print(X)
X = X.to_numpy()
y_train = info['y_train']



df_splits = [X_train, X_test]
#X = np.concatenate(df_splits, axis=0)
#X_embedded = TSNE(learning_rate='auto', init='pca', random_state=seed).fit_transform(X)
# take 1000 random indices
positive_indices = []
negative_indices = []
counts_positive_class = 0
counts_negative_class = 0
for y_index, y in enumerate(y_train):
    if y == 1:
        if counts_positive_class < 500:
            counts_positive_class += 1
            positive_indices.append(y_index)
    elif y == 0:
        if counts_negative_class < 500:
            counts_negative_class += 1
            negative_indices.append(y_index)
    if counts_positive_class == 500 and counts_negative_class == 500:
        break

positive_indices = np.array(positive_indices)
negative_indices = np.array(negative_indices)
X_train = X_train.to_numpy()
plt.scatter(X_train[positive_indices, 2], X_train[positive_indices, 3], label='positive class', color='red')
plt.scatter(X_train[negative_indices, 2], X_train[negative_indices, 3], label='negative class', color='blue')
plt.xlabel('Capital Gain')
plt.ylabel('Education Num.')
plt.legend()
plt.savefig('importances.pdf', bbox_inches='tight')
