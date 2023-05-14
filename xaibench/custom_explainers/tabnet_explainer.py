import numpy as np
from tqdm import tqdm

class TabNetExplainer:
    """Interpretable neural network interface.

    Parameters
    ----------
    model : function or iml.Model
        User supplied function that takes a matrix of samples (# samples x # features) and
        computes a the output of the model for those samples. The output can be a vector
        (# samples) or a matrix (# samples x # model outputs).

    data : numpy.array
        The background dataset.

    mode : "classification" or "regression"
        Control the mode of LIME tabular.
    """

    def __init__(self, model, data, mode="classification"):
        self.model = model
        assert mode in ["classification", "regression"]
        self.mode = mode

        if str(type(data)).endswith("pandas.core.frame.DataFrame'>"):
            data = data.values
        self.data = data
        self.explainer = self.model

        out = self.model.predict(data[0:1])
        if len(out.shape) <= 1:
            self.out_dim = 1
            self.flat_out = True
            if mode == "classification":

                def pred(X):  # assume that 1d outputs are probabilities
                    preds = self.model.predict(X).reshape(-1, 1)
                    p0 = 1 - preds
                    return np.hstack((p0, preds))

                self.model = pred
        else:
            self.out_dim = self.model.predict(data[0:1]).shape[1]
            self.flat_out = False

    def attributions(self, X, nsamples=5000, num_features=None):

        self.model.predict(X)
        weights, _ = self.model.explain(X)

        return weights

class TabNetExplainerWrapper:
    def __init__(self, f, X, **kwargs):
        self.f = f
        self.X = X
        self.explainer = TabNetExplainer(self.f, self.X, mode="regression", **kwargs)

    def explain(self, x):
        x = x.to_numpy()
        weight_values = self.explainer.attributions(x)
        self.expected_values = np.zeros(
            x.shape[0]
        )
        weight_values = weight_values

        return weight_values * x