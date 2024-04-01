import numpy as np


def rmse(y_true, y_pred):
    """
    Calculate the Root Mean Square Error (RMSE) between the true and predicted values.

    Parameters
    ----------
    y_true : numpy.ndarray
        A 1D array of the true target values.
    y_pred : numpy.ndarray)
        A 1D array of the predicted target values.

    Returns
    -------
    float
        The RMSE between the true and predicted values.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """
    Calculate the Mean Absolute Error (MAE) between the true and predicted values.

    Parameters
    ----------
    y_true : numpy.ndarray
        A 1D array of the true target values.
    y_pred : numpy.ndarray)
        A 1D array of the predicted target values.

    Returns
    -------
    float
        The MAE between the true and predicted target values.
    """
    return np.mean(np.abs(y_true - y_pred))


def coverage(y_true, y_pred_lower, y_pred_upper):
    """
    Calculate the coverage of the prediction intervals.

    Parameters
    ----------
    y_true : numpy.ndarray
        A 1D array of the true target values.
    
    y_pred_lower : numpy.ndarray
        A 1D array of the lower bounds of the predicted intervals.

    y_pred_upper : numpy.ndarray
        A 1D array of the upper bounds of the predicted intervals.

    Returns
    -------
    float
        The proportion of true values that fall within the predicted intervals.
    """
    coverage = (y_true >= y_pred_lower) & (y_true <= y_pred_upper)
    return np.mean(coverage)


def sharpness(y_pred_lower, y_pred_upper):
    """
    Calculate the sharpness of the prediction intervals.

    Parameters
    ----------
    y_pred_lower : numpy.ndarray
        A 1D array of the lower bounds of the predicted intervals.

    y_pred_upper : numpy.ndarray
        A 1D array of the upper bounds of the predicted intervals.

    Returns
    -------
    float
        The average width of the prediction intervals.
    """
    return np.mean(y_pred_upper - y_pred_lower)