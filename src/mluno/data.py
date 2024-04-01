import numpy as np


def make_line_data(n_samples=100, beta_0=0,beta_1=1, sd = 1,X_low=-10, X_high=10, random_seed=None):

    """
    Generate line data for a simple linear regression model with noise.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.

    beta_0 : float
        The true intercept of the linear model.

    beta_1 : float
        The true slope of the linear model.

    sd : float
        Standard deviation of the normally distributed errors.

    X_low : float
        Lower bound for the uniform distribution of X.

    X_high : float
        Upper bound for the uniform distribution of X.

    random_seed : int
        Seed to control randomness.

    Returns
    -------
    tuple
        A tuple containing the X and y arrays. X is a 2D array with shape (n_samples, 1) and y is a 1D array with shape (n_samples,). X contains the simulated X values and y contains the corresponding true mean of the linear model with added normally distributed errors.
    """

    np.random.seed(random_seed)

    X = np.random.uniform(X_low, X_high, size = (n_samples, 1))
    y = beta_0 + beta_1 * X.ravel() + np.random.normal(scale=sd, size=n_samples)
    return X,y


def make_sine_data(n_samples=100, sd=1, X_low=-6, X_high=6, random_seed=None):
    """
    Generate sine data with noise.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.

    sd : float
        Standard deviation of the normally distributed errors.

    X_low : float
        Lower bound for the simulated distribution of X.

    X_high : float
        Upper bound for the simulated distribution of X.

    random_seed : int
        Seed to control randomness.

    Returns
    -------
    tuple
        A tuple containing the X and y arrays. X is a 2D array with shape (n_samples, 1) and y is a 1D array with shape (n_samples,). X contains the simulated X values and y contains the corresponding sine values with added normally distributed errors.
    """

    np.random.seed(random_seed)

    X = np.random.uniform(X_low, X_high, size=(n_samples, 1))
    y = np.sin(X).ravel() + np.random.normal(scale=sd, size=n_samples)
    return X, y


# The split train and test data: (X_train, X_test, y_train, y_test).
def split_data(X, y, holdout_size=0.2, random_seed=None):
    """
    Split the data into training and test sets.

    Parameters
    ----------
    X : numpy.ndarray
        The input features.

    y : numpy.ndarray
        The target variable.

    holdout_size : float
        The proportion of the data to include in the test split.
        
    random_seed : int
        Seed to control randomness.

    Returns
    -------
    tuple
        A tuple containing the split training and test data: (X_train, X_test, y_train, y_test).
    """

    np.random.seed(random_seed)

    n_samples = X.shape[0]
    n_holdout = int(n_samples * holdout_size)

    indices = np.random.permutation(n_samples)

    holdout_indices = indices[:n_holdout]
    train_indices = indices[n_holdout:]

    X_train, X_test = X[train_indices], X[holdout_indices]
    y_train, y_test = y[train_indices], y[holdout_indices]

    return X_train, X_test, y_train, y_test