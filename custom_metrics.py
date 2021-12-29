import numpy as np


def aic(mse, n_data, n_params):
    return n_data * np.log(mse) + 2 * n_params


def aicc(mse, n_data, n_params):
    return n_data * np.log(mse) + (n_data + n_params) / (1 - (n_params + 2) / n_data)


def bic(mse, n_data, n_params):
    return n_data * np.log(mse) + n_params * np.log(n_data)

