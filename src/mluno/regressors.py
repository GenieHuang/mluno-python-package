import numpy as np

class KNNRegressor:
    """
    A class used to represent a K-Nearest Neighbors Regression model.

    ...

    Attributes
    ----------
    k : int
        The number of nearest neighbors to consider for regression.
    X : numpy.ndarray
        The feature data used for training the model.
    y : numpy.ndarray
        The target data used for training the model.

    Methods
    -------
    fit(X, y):
        Fit the model using X as training data and y as target values.
    predict(X_new):
        Predict the target for the provided data.
    _predict(x_new):
        Predict the target for a single instance.
    """

    def __init__(self, k=5):
        """
        Parameters
        ----------
        k : int, optional
            The number of nearest neighbors to consider for regression (default is 5).
        """
        self.k = k

    def fit(self, X, y):
        """
        Fit the model using X as training data and y as target values.

        Parameters
        ----------
        X : numpy.ndarray
            The feature data used for training the model.
        y : numpy.ndarray
            The target data used for training the model.
        """

        self.X = X
        self.y = y

    def __repr__(self) -> str:
        """
        Returns a string representation of the KNNRegressor model.

        Returns
        -------
        str
            A string representation of the KNNRegressor model.
        """

        return f"KNN Regression model with k = {self.k}."

    def predict(self, X_new):
        """
        Predict the target for the provided data.

        Parameters
        ----------
        X_new : numpy.ndarray
            The feature data for which to predict targets.

        Returns
        -------
        numpy.ndarray
            The predicted targets for the provided data.
        """

        predicted_labels = [self._predict(x) for x in X_new]
        return np.array(predicted_labels)

    def _predict(self, x_new):
        """
        Predict the target for a single instance.

        Parameters
        ----------
        x_new : numpy.ndarray
            The feature data for which to predict the target.

        Returns
        -------
        float
            The predicted target for the provided data.
        """

        # compute distances between new x and all samples in the X data
        distances = [np.linalg.norm(x_new - x) for x in self.X]
        # sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[: self.k]
        # extract the labels of the k nearest neighbor training samples
        k_nearest_y = self.y[k_indices]
        # return the mean of the k nearest neighbors
        return np.mean(k_nearest_y)
    

# regressors.LinearRegressor(self), A class used to represent a Simple Linear Regressor., attributes: weights: The weights of the linear regression model. Here, the weights are represented by the vector which for univariate regression is a 1D vector of length two, 

class LinearRegressor:
    """
    A class used to represent a Simple Linear Regressor.

    ...

    Attributes
    ----------
    weights : numpy.ndarray
        The weights of the linear regression model. Here, the weights are represented by the vector which for univariate regression is a 1D vector of length two.

    Methods
    -------
    fit(X, y):
        Fit the model using X as training data and y as target values.
    predict(X_new):
        Predict the target for the provided data.
    """

    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        """
        Fit the model using X as training data and y as target values.

        Parameters
        ----------
        X : numpy.ndarray
            The feature data used for training the model.
        y : numpy.ndarray
            The target data used for training the model.
        """

        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # calculate the weights
        self.weights = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def predict(self, X):
        """
        Predict the target for the provided data.

        Parameters
        ----------
        X : numpy.ndarray
            The feature data for which to predict targets.

        Returns
        -------
        numpy.ndarray
            The predicted targets for the provided data.
        """

        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        return X_b.dot(self.weights)
        