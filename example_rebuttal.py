import sklearn.linear_model
import numpy as np

nr_examples = 10000

labels = np.arange(nr_examples)

x_1 = labels / 10

logistic = sklearn.linear_model.LinearRegression()

# random noise for x_2 and x_3
x_2 = np.random.rand(nr_examples)
x_3 = np.random.rand(nr_examples)

X = np.array([x_1, x_2, x_3]).T

logistic.fit(X, labels)

# First scenario
print("First scenario:")
print(logistic.coef_)

logistic = sklearn.linear_model.LinearRegression()

x_2 = labels
x_3 = -labels

X = np.array([x_1, x_2, x_3]).T

logistic.fit(X, labels)

print("Second scenario:")
print(logistic.coef_)
