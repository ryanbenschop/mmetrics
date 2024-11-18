import numpy as np
from varmodel import VARModel

"""
The following code provides a class designed for implementing the information criteria based lag order selection procedure for a VAR model.
"""

class VARLagOrderSelector:
    def __init__(self):
        """
        Initialiser for the VARLagOrderSelector class.
        """
        self.data = None
        self.var_models = None

        self.aic_values = None
        self.hqc_values = None
        self.bic_values = None
    
    def compute_lag_order_statistics(self, data: np.ndarray, max_lag: int):
        """
        Computes the AIC, HQC, and BIC values for a range of lag orders for a VAR model.

        Args:
            data (np.ndarray): The data to be used for the VAR model.
            max_lag (int): The maximum lag order to be considered.
        """
        var_models = []

        aic_values = []
        hqc_values = []
        bic_values = []

        for num_lags in range(1, max_lag + 1):
            var_model = VARModel()
            var_model.estimate_ols(data = data[max_lag - num_lags:, :], num_lags = num_lags)
            var_models.append(var_model)

            fit_term = np.log(np.linalg.det(var_model.Σ_ε))
            num_regressors = var_model.num_vars * (num_lags + 1)

            aic_penalty = 2 / var_model.num_obs
            hqc_penalty = 2 * np.log(np.log(var_model.num_obs)) / var_model.num_obs
            bic_penalty = np.log(var_model.num_obs) / var_model.num_obs

            aic = fit_term + aic_penalty * num_regressors
            aic_values.append(aic)

            hqc = fit_term + hqc_penalty * num_regressors
            hqc_values.append(hqc)

            bic = fit_term + bic_penalty * num_regressors
            bic_values.append(bic)
        
        self.data = data
        
        self.var_models = var_models
        self.aic_values = aic_values
        self.hqc_values = hqc_values
        self.bic_values = bic_values