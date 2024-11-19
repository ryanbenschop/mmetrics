import numpy as np
from varmodel import VARModel

class AutocovarianceModel:
    def __init__(self):
        self.data = None

        self.autocovariances = None
        self.autocorrelations = None
    
    def construct_F_h(self, T: int, h: int):
        F_h = np.vstack([
            np.zeros((h, T)),
            np.hstack([
                np.eye(T - h), np.zeros((T - h, h))
            ])
        ])

        return F_h

    def estimate_autocovariance(self, data: np.ndarray, h: int):
        num_obs = np.shape(data)[0]
        F_h = self.construct_F_h(num_obs, h)
        C_h = data.T @ F_h @ data / num_obs

        return C_h
    
    def estimate_autocorrelation(self, data: np.ndarray, h: int):
        num_obs = np.shape(data)[0]
        C_0 = self.estimate_autocovariance(data, 0)
        C_h = self.estimate_autocovariance(data, h)

        D = np.diag(np.sqrt(C_0))
        D = np.diag(D)
        D_inv = np.linalg.inv(D)

        R_h = D_inv @ C_h @ D_inv

        return R_h

def estimate_residual_autocorrelations(var_model: VARModel, max_lag: int):
    num_obs = var_model.num_obs
    num_vars = var_model.num_vars
    num_lags = var_model.num_lags

    resid_autocovariance_model = AutocovarianceModel()

    R = [resid_autocovariance_model.estimate_autocorrelation(var_model.resids.T, h) for h in range(1, max_lag + 1)]

    Σ_ε = var_model.Σ_ε * (num_obs - num_vars * num_lags - 1) / num_obs

    C_0 = resid_autocovariance_model.estimate_autocovariance(var_model.resids.T, 0)
    # D = np.diag(np.sqrt(Σ_ε))
    D = np.diag(np.sqrt(C_0))
    D = np.diag(D)
    D_inv = np.linalg.inv(D)

    R_u_hat = D_inv @ Σ_ε @ D_inv

    Z = var_model.Z

    Γ_hat = Z @ Z.T / num_obs
    Γ_hat_inv = np.linalg.inv(Γ_hat)

    std_errs = []

    MA_selection_mat = np.column_stack((
            np.eye(num_vars),
            np.zeros((num_vars, num_vars * (num_lags - 1)))
        ))

    for h in range(1, max_lag + 1):
        # Δ_mats = [MA_selection_mat @ np.linalg.matrix_power(var_model.Γ, i) @ MA_selection_mat.T for i in range(h - num_lags, h)]
        Δ_mats = [MA_selection_mat @ np.linalg.matrix_power(var_model.Γ, i) @ MA_selection_mat.T for i in range(h - 1, h - num_lags - 1, -1)]
        # Set Δ to be a column of zeros of size num_vars by 1, then append all the Δ_mats
        Δ = np.row_stack((
            np.zeros((1, num_vars)),
            np.vstack(Δ_mats)
        ))

        # print(np.shape(Δ))

        Σ_R_h = np.kron(
            R_u_hat - D_inv @ Σ_ε @ Δ.T @ Γ_hat_inv @ Δ @ Σ_ε @ D_inv,
            R_u_hat
        )

        std_errs_h = np.sqrt(np.diag(Σ_R_h)) / np.sqrt(num_obs)
        std_errs_h = std_errs_h.reshape((num_vars, num_vars))
        std_errs_h = std_errs_h.T
        std_errs.append(std_errs_h)

    return R, std_errs