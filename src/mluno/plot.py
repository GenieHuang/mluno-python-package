import numpy as np
import matplotlib.pyplot as plt


def plot_predictions(X, y, regressor, conformal=False, title=''):
    """
    Plot predictions of a regressor along with the data.

    Parameters
    ----------
    X : `ndarray`
        A 2D array of the input data.

    y : `ndarray`
        A 1D array of the same length as `X` of the true target values.

    regressor : `object`
        A regressor object with a `predict` method that can be used to predict target values.

    conformal : `bool`
        If True, the regressor is assumed to return prediction intervals (lower and upper bounds) along with the predictions. The prediction intervals are plotted as a shaded area.

    title : `str`
        The title of the plot.

    Returns
    -------
    `matplotlib.figure.Figure`
        The figure object for the plot.

    `matplotlib.axes.Axes`
        The axes object for the plot.
    
    Notes
    -----
    This function assumes that the `predict` method of `regressor` returns a tuple of three elements (predictions, lower bounds, upper bounds) when `conformal` is True.
    """
    fig, ax = plt.subplots()
    
    if conformal:
        y_pred, y_pred_lower, y_pred_upper = regressor.predict(X)
        ax.fill_between(X.ravel(), y_pred_lower, y_pred_upper, color="lightblue", alpha=0.2)

    else:
        y_pred = regressor.predict(X)
    
    ax.plot(X, y_pred, color="darkblue")
    ax.set_title(title)

    return fig, ax