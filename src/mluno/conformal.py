import numpy as np


class ConformalPredictor:
    """
    A class used to represent a Conformal Predictor.

    Parameters
    ----------
    regressor : `object`
        The regression model to be used for prediction.
    alpha : `float`
        The significance level used for prediction interval calculation.
    
    Attributes
    ----------
    scores : `ndarray`
        The predicted scores from the fitted regression model.
    quantile : `float`
        The quantile value calculated based on the scores and alpha.

    """

    def __init__(self, regressor, alpha=0.05):
        
        self.regressor = regressor
        self.alpha = alpha
        self.scores = None
        self.quantile = None

    def fit(self, X, y):
        """
        Fit the model using X as training data and y as target values.

        Parameters
        ----------
        X : `ndarray`
            The feature data used for training the model.
        y : `ndarray`
            The target data used for training the model.
        """
        self.regressor.fit(X, y)
        self.scores = np.abs(self.regressor.predict(X) - y)
        self.quantile = np.quantile(self.scores, 1 - self.alpha)

    def predict(self, X):
        """
        Predict the target and prediction interval for the provided data.

        Parameters
        ----------
        X : `ndarray`
            The feature data for which to predict targets.

        Returns
        -------
        `tuple`
            The predicted target (y_pred, 1D `ndarray`), lower bound of prediction interval (y_lower, 1D `ndarray`), and upper bound of prediction interval (y_upper, 1D `ndarray`).
        """
        
        y_pred = self.regressor.predict(X)
        y_lower = y_pred - self.quantile
        y_upper = y_pred + self.quantile
        return y_pred, y_lower, y_upper