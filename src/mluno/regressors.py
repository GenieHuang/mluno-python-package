import numpy as np

class KNNRegressor:
    """
    A class used to represent a K-Nearest Neighbors Regression model.

    Parameters
    ----------
    k : `int`
        The number of nearest neighbors to consider for regression.
    """

    def __init__(self, k=5):

        self.k = k

    def fit(self, X, y):
        """
        Fit the model using `X` as training data and `y` as target values.

        Parameters
        ----------
        X : `ndarray`
            The feature data used for training the model, which is a 2D array of shape `(n_samples, 1)`.

        y : `ndarray`
            The target data used for training the model, which is a 1D array of shape `(n_samples,)`.
        """

        self.X = X
        self.y = y

    def __repr__(self) -> str:

        return f"KNN Regression model with k = {self.k}."

    def predict(self, X_new):
        """
        Predict the target for the provided data.

        Parameters
        ----------
        X_new : `ndarray`
            The feature data for which to predict targets, which is a 2D array of shape `(n_samples, 1)`.

        Returns
        -------
        `ndarray`
            The predicted targets for the provided data, which is a 1D array of shape `(n_samples,)`.
        """

        predicted_labels = [self._predict(x) for x in X_new]
        return np.array(predicted_labels)

    def _predict(self, x_new):

        # compute distances between new x and all samples in the X data
        distances = [np.linalg.norm(x_new - x) for x in self.X]
        # sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[: self.k]
        # extract the labels of the k nearest neighbor training samples
        k_nearest_y = self.y[k_indices]
        # return the mean of the k nearest neighbors
        return np.mean(k_nearest_y)
    


class LinearRegressor:
    """
    A class used to represent a Simple Linear Regressor.

    Attributes
    ----------
    weights : `ndarray`
        The weights learned by the model.

    """

    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        """
        Fit the model using X as training data and y as target values.

        Parameters
        ----------
        X : `ndarray`
            The feature data used for training the model, which is a 2D array of shape `(n_samples, 1)`.
        
        y : `ndarray`
            The target data used for training the model, which is a 1D array of shape `(n_samples,)`.
        """

        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # calculate the weights
        self.weights = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def predict(self, X):
        """
        Predict the target for the provided data.

        Parameters
        ----------
        X : `ndarray`
            The feature data for which to predict targets, which is a 2D array of shape `(n_samples, 1)`.

        Returns
        -------
        `ndarray`
            The predicted targets for the provided data, which is a 1D array of shape `(n_samples,)`.
        """

        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        return X_b.dot(self.weights)