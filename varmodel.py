import numpy as np

"""
The following code provides a class designed for estimating and analysing vector autoregressive (VAR) models. The model can be written in the form:
y_t = μ + \sum_{l = 1}^p Φ_l y_{t - l} + ε_t
where y_t consists of a set of k dependent variables, μ is a constant in R^k, Φ_1, ..., Φ_p are k by k matrices, and ε_t ~ (0, Σ) consists of k reduced form shocks.

Some other useful representations of the model include:

-> the compact form:
y_t = Φ_+ z_{t - 1} + ε_t
where z_{t - 1} := (1, y_{t - 1}, ..., y_{t - p}) and Φ_+ := [μ Φ_1 ... Φ_p].

-> the companion form:
Y_t = ν + F_Φ Y_{t - 1} + E_t
where Y_t := (y_t, y_{t-1}, ..., y_{t - p + 1}) (k * p by 1), ν := (μ, 0, ..., 0) (k * p by 1), E_t = [ε_t, 0, ..., 0] (k * p by 1), and
F_Φ := [
    Φ_1 ... Φ_{p - 1}   Φ_p
    I_k ... 0           0
    ⋮   ⋮   ⋮            ⋮
    0   ... I_k         0
] (k * p by k * p).

-> the (more compact) companion form:
Y_t = Γ Z_{t - 1} + E_t
where Γ = [ν F_Φ] and Z_{t - 1} = (1, Y_{t - 1}).

-> the data matrix form:
Y = Φ_+ Z + E
where Y, Z, and E are matrices of observations on y_t, z_{t - 1}, and the corresponding unobserved errors.
"""
class VARModel:
    def __init__(self):
        """
        Initialiser for the VARModel class.
        This will store the attributes for the model, including the model parameters, data matrices for estimation, and any other relevant matrices for
        computations.
        """
        # Model parameters
        self.μ = None                   # Constant term in the standard VAR representation
        self.Φ_mats = None              # Autoregressive coefficients in the standard VAR representation

        self.Φ_plus = None              # Constant term and autoregressive coefficients in the compact VAR

        self.ν = None                   # Constant term in the companion form
        self.F_Φ = None                 # Autoregressive coefficients in the companion form

        self.Γ = None                   # Constant term and autoregressive coefficients in the compact companion form

        self.Σ_ε = None                 # Error covariance matrix

        # Model specification variables
        self.num_lags = None            # Number of lags, p
        self.num_vars = None            # Number of endogenous variables, k

        # Data matrices
        self.estimation_data = None     # Data for estimation
        self.num_obs = None             # Number of observations on each variable (after dropping the initial conditions from the original data)

        self.Y = None                   # Matrix of regressands
        self.Z = None                   # Matrix of regressands
        self.resids = None              # Matrix of residuals

        self.ZZt_inv = None             # (ZZ')^{-1}

        # Matrices for estimation results
        self.coefficient_asy_cov = None # Estimated asymptotic covariance matrix of the estimator of Φ_+
        self.t_stat_mat = None          # Matrix of t statistics for the estimator of Φ_+

    #----------------------- Handle the updating of model parameters and specification variables
    def check_Φ_plus_dims(self, Φ_plus: np.ndarray):
        """
        Determines if a given candidate for Φ_+ is valid.

        Args:
            Φ_+ (np.ndarray): A numpy array corresponding with the constant term and autoregressive coefficients in the compact VAR.
        """
        # Check that it is two dimensional
        assert len(Φ_plus.shape) == 2, "Φ_+ should be a 2D array."

        # Check that the shape is k by kp + 1
        assert Φ_plus.shape[1] % Φ_plus.shape[0] == 1, "Dimensions of Φ_+ are invalid; expected a k by kp + 1 matrix."
    
    def check_Σ_ε_dims(self, Σ_ε: np.ndarray):
        """
        Determines if a given candidate for Σ_ε is valid.

        Args:
            Σ_ε (np.ndarray): A numpy array corresponding with the error covariance matrix.
        """
        # Check that it is two dimensional
        assert len(Σ_ε.shape) == 2, "Σ_ε should be a 2D array."

        # Check that it is square
        assert Σ_ε.shape[0] == Σ_ε.shape[1], "Dimensions of Σ_ε are invalid; expected a square matrix."

    def set_parameters(self, Φ_plus: np.ndarray, Σ_ε: np.ndarray):
        """
        Update the model parameters.

        Args:
            Φ_+ (np.ndarray): A numpy array corresponding with the constant term and autoregressive coefficients in the compact VAR.
            Σ_ε (np.ndarray): A numpy array corresponding with the error covariance matrix.
        """
        # Check that the dimensions of Φ_+ are valid
        self.check_Φ_plus_dims(Φ_plus)

        # Check that the dimensions of Σ_ε are valid
        self.check_Σ_ε_dims(Σ_ε)

        # Retrieve the number of variables
        num_vars = Φ_plus.shape[0]

        # Retrieve the number of lags
        num_lags = int((Φ_plus.shape[1] - 1) / num_vars)

        # Retrieve the parameters for the standard representation
        μ, Φ_mats = self.retrieve_standard_form_mats(Φ_plus, num_vars, num_lags)

        # Retrieve the companion form matrices
        ν, F_Φ, Γ = self.retrieve_companion_form_mats(Φ_plus, num_vars, num_lags)

        # Update the class attributes
        self.Φ_plus = Φ_plus
        self.Σ_ε = Σ_ε

        self.μ = μ
        self.Φ_mats = Φ_mats

        self.ν = ν
        self.F_Φ = F_Φ

        self.Γ = Γ

        self.num_lags = num_lags
        self.num_vars = num_vars
    
    def retrieve_standard_form_mats(self, Φ_plus: np.ndarray, num_vars: int = None, num_lags: int = None):
        """
        Retrieve the parameter matrices for the standard form of the VAR.

        Args:
            Φ_+ (np.ndarray): A numpy array corresponding with the constant term and autoregressive coefficients in the compact VAR.

        Returns:
            μ (np.ndarray): A numpy array corresponding with the constant term in the standard form of the VAR.
            Φ_mats (np.ndarray): An array of numpy arrays where each element is the corresponding matrix Φ_l in the standard form of the VAR.
        """
        self.check_Φ_plus_dims(Φ_plus)

        # Retrieve the number of variables if it was not specified
        if num_vars is None:
            num_vars = Φ_plus.shape[0]

        # Retrieve the number of lags if it was not specified
        if num_lags is None:
            num_lags = int((Φ_plus.shape[1] - 1) / num_vars)
        
        # Retrieve the parameters for the standard representation
        μ = Φ_plus[:, 0]
        Φ_mats = [Φ_plus[:, 1 + i * num_vars: 1 + (i + 1) * num_vars] for i in range(num_lags)]

        return μ, Φ_mats
    
    def retrieve_companion_form_mats(self, Φ_plus: np.ndarray, num_vars: int = None, num_lags: int = None):
        """
        Retrive the parameter matrices for the companion form of the VAR.

        Args:
            Φ_+ (np.ndarray): A numpy array corresponding with the constant term and autoregressive coefficients in the compact VAR.

        Returns:
            ν (np.ndarray): A numpy array corresponding with the constant term in the companion form of the VAR.
            F_Φ (np.ndarray): A numpy array corresponding with the autoregressive term in the companion form of the VAR.
            Γ: A numpy array corresponding with the constant and autoregressive term in the compact companion form of the VAR.
        """
        self.check_Φ_plus_dims(Φ_plus)

        # Retrieve the number of variables if it was not specified
        if num_vars is None:
            num_vars = Φ_plus.shape[0]

        # Retrieve the number of lags if it was not specified
        if num_lags is None:
            num_lags = int((Φ_plus.shape[1] - 1) / num_vars)

        # Construct the ν and F_Φ matrices from the companion form //// TO DO
        ν, F_Φ = None, None

        # Construct the Γ matrix from the compact companion form
        Γ = np.row_stack((
                Φ_plus[:, 1:],
                np.column_stack((np.eye(num_vars * (num_lags - 1)), np.zeros((num_vars * (num_lags - 1), num_vars))))
            ))

        return ν, F_Φ, Γ

    #----------------------- Least squares estimation of the parameters
    def extract_regression_matrices(self, data: np.ndarray, num_lags: int, const: bool):
        """
        Construct the regressand and regressor matrices, Y and Z, corresponding with the data matrix representation of the VAR.

        Args:
            data (np.ndarray): A numpy array of observations on the endogenous model variables. The rows represent time periods, and the columns represent the different
                                variables.
            num_lags (int): The number of lags to include in the VAR model.
            const (bool): Set to true if a constant should be included in the model.
        
        Returns:
            Y (np.ndarray): The matrix of regressands.
            Z (np.ndarray): The matrix of regressors.
        """
        # Check that the model specification is valid
        assert num_lags > 0 or const == True, "Unable to construct data matrices. The model should either include a constant or at least one lag."

        # Get the number of observations
        num_obs = np.shape(data)[0] - num_lags

        # Ensure there are enough observations
        assert num_obs > 0, "Insufficient observations to estimate the model. There should be at least p + 1 observations."
        
        # Construct the regressand matrix
        Y = data[num_lags:, :].T

        # Construct the regressor matrix
        ones_col = np.ones(num_obs)
        Z = np.column_stack(
            [
                data[i:i - num_lags, :]
            for i in range(num_lags - 1, -1, -1)]
        ) if num_lags > 0 else ones_col

        if const and num_lags > 0:
            Z = np.column_stack((ones_col, Z)).T
        
        return Y, Z


    def compute_ols_coeffs(self, Y: np.ndarray, Z: np.ndarray, ZZt_inv: np.ndarray = None):
        """
        Compute the least squares coefficient estimates of Φ_+.

        Args:
            Y (np.ndarray): The matrix of regressands.
            Z (np.ndarray): The matrix of regressors.
            ZZt_inv (np.ndarray): (ZZ')^{-1}. May be precomputed for efficiency.
        
        Returns:
            Φ_plus_ols (np.ndarray): The estimates of Φ_+.
        """
        # Compute ZZt_inv if it was not specified
        if ZZt_inv is None:
            ZZt_inv = np.linalg.inv(Z @ Z.T)

        # Compute the least squares coefficient estimates
        Φ_plus_ols = Y @ Z.T @ ZZt_inv

        return Φ_plus_ols
    
    def estimate_resid_cov(self, resids: np.ndarray, num_obs: int, num_vars: int, num_lags: int, cov_type: str):
        """
        Compute the estimate of the residual covariance matrix.

        Args:
            resids (np.ndarray): A numpy array containing the residuals.
            num_obs (int): The number of observations of the regressands.
            num_vars (int): The number of variables included in the model.
            num_lags (int): The number of lags included in the model.
            cov_type (str): The type of covariance matrix estimator to use. Should be either "mm" or "db".capitalize
        
        Returns:
            np.ndarray: The estimate of the residual covariance matrix.
        """
        # Compute the method of moments/ML estimator of the error covariance
        Σ_ε_mm = resids @ resids.T / num_obs

        # Return Σ_ε_mm if covariance type is mm; otherwise, adjust for the bias and return
        if cov_type == "mm":
            return Σ_ε_mm
        else:
            Σ_u_db = Σ_ε_mm * num_obs / (num_obs - num_vars * num_lags - 1)     # Debiased estimator
            return Σ_u_db
    
    def estimate_ols(self, data: np.ndarray, num_lags: int, const: bool = True, cov_type: str = "db"):
        """
        Estimate the VAR model using ordinary least squares. Update the model parameters and store the results.

        Args:
            data (np.ndarray): A numpy array of observations on the endogenous model variables. The rows represent time periods, and the columns represent the different
                                variables.
            num_lags (int): The number of lags to include in the VAR model.
            const (bool, optional): Set to true if a constant should be included in the model. Defaults to True.
            cov_type (str, optional): The type of covariance matrix estimator to use. Should be either "mm" or "db". Defaults to "db".
        """
        # Retrieve the model specification variables
        num_obs, num_vars = np.shape(data)
        num_obs -= num_lags

        # Extract the matrices of regressands and regressors
        Y, Z = self.extract_regression_matrices(data, num_lags, const)
        ZZt_inv = None

        try:
            ZZt_inv = np.linalg.inv(Z @ Z.T)
        except np.linalg.LinAlgError:
            print("Error: The regressor matrix is singular.")

        # Compute the estimate of Φ_+
        Φ_plus_ols = self.compute_ols_coeffs(Y, Z, ZZt_inv)

        # Compute the residuals
        resids = Y - Φ_plus_ols @ Z

        # Compute the estimate of the error covariance matrix
        Σ_ε_est = self.estimate_resid_cov(resids, num_obs, num_vars, num_lags, cov_type)

        # Compute the estimator of the asymptotic variance of the coefficients
        coefficient_asy_cov_est = self.estimate_coefficient_asy_cov(Z, Σ_ε_est)

        # Compute the matrix of t statistics
        t_stat_mat = self.compute_t_stats(Φ_plus_ols, coefficient_asy_cov_est)

        # Update the class attributes
        self.set_parameters(Φ_plus_ols, Σ_ε_est)

        self.num_lags = num_lags
        self.num_vars = num_vars

        self.estimation_data = data
        self.num_obs = num_obs

        self.Y = Y
        self.Z = Z
        self.ZZt_inv = ZZt_inv
        self.resids = resids

        self.coefficient_asy_cov = coefficient_asy_cov_est
        self.t_stat_mat = t_stat_mat
    
    def estimate_coefficient_asy_cov(self, Z: np.ndarray, resid_cov_matrix: np.ndarray):
        """
        Compute the estimated asymptotic covariance matrix of the estimator of Φ_+.

        Args:
            Z (np.ndarray): The matrix of regressors.
            resid_cov_matrix (np.ndarray): The estimated covariance matrix of the residuals.

        Returns:
            np.ndarray: The estimated asymptotic covariance matrix of the estimator of Φ_+.
        """
        # Compute the estimated asymptotic covariance matrix of the estimator of Φ_+
        asy_cov_est = np.kron(np.linalg.inv(Z @ Z.T), resid_cov_matrix)

        return asy_cov_est
    
    def compute_t_stats(self, Φ_plus: np.ndarray, coefficient_asy_cov: np.ndarray):
        """
        Compute the matrix of t statistics for the estimator of Φ_+.

        Args:
            Φ_plus (np.ndarray): The estimate of Φ_+.
            coefficient_asy_cov (np.ndarray): The estimated asymptotic covariance matrix of the estimator of Φ_+.
        
        Returns:
            np.ndarray: The matrix of t statistics for the estimator of Φ_+.
        """
        # Initialise the t statistics matrix
        t_stat_mat = Φ_plus.copy()

        # Compute the t statistics
        counter = 0
        num_rows, num_cols = np.shape(Φ_plus)

        for j in range(num_cols):
            for i in range(num_rows):
                t_stat_mat[i][j] /= np.sqrt(coefficient_asy_cov[counter][counter])
                counter += 1
        
        return t_stat_mat
    
    #----------------------- Impulse response analysis
    def compute_IRFs(self, horizon: int, num_lags: int, J: np.ndarray = None):
        """
        Compute the reduced form impulse response function matrix for a given horizon.

        Args:
            horizon (int): The forecast horizon for the IRFs.
            num_lags (int): Number of lags in the VAR model.
            num_vars (int): Number of endogenous variables in the model.
            J (np.ndarray, optional): Selection matrix that isolates contemporaneous impacts. 
                                      Defaults to None, in which case a standard selection matrix is created.

        Returns:
            np.ndarray: Reduced-form IRF matrix at the specified horizon.
        """
        assert Γ is not None, "Unable to compute reduced form IRFs, model parameters are undefined."
        Γ = self.Γ

        # Construct the selection matrix if none is provided
        if J is None:
            J = np.column_stack((
                np.eye(num_vars),
                np.zeros((num_vars, num_vars * (num_lags - 1)))
            ))

        # Compute the reduced-form IRF at the specified horizon
        Ξ_h = J @ np.linalg.matrix_power(Γ, horizon) @ J.T

        return Ξ_h
    
    #----------------------- Forecasting
    def forecast(self, horizon: int, init_conditions: np.ndarray = None, start_period: int = None):
        """
        Generate h step ahead forecasts from the VAR model. The inital conditions may be provided explicitly, or the starting period may be specified. If neither
        is provided, the forecast will be generated from the last observation in the data.

        Args:
            horizon (int): The forecast horizon.
            init_conditions (np.ndarray, optional): The initial conditions for the forecast. Defaults to None.
            start_period (int, optional): The starting period for the forecast. Defaults to None.

        Returns:
            np.ndarray: The array of point forecasts at the specified horizon.
            np.ndarray: The estimated forecast MSE.
        """
        # Check that if the initial conditions are not specified, the data matrix has been provided
        assert not (init_conditions is None and self.estimation_data is None), "Unable to produce forecasts. Neither the model data nor initial conditions have been provided."

        # Check that both initial condittions and start_period were not provided
        assert not (init_conditions is not None and start_period is not None), "Unable to produce forecasts. \
            Only one of init_conditions and start_period should be specified."

        # If no initial conditions were provided and no start period was provided, set the start period to the end of the sample
        if init_conditions is None and start_period is None:
            start_period = self.num_obs

        # Construct the selection matrix for computing the forecasts
        forecast_selection_mat = np.column_stack((
            np.zeros(self.num_vars),
            np.eye(self.num_vars),
            np.zeros((self.num_vars, self.num_vars * (self.num_lags - 1)))
        ))

        # Construct the auxiliary parameter matrix for computing the forecasts
        forecast_aux_mat = np.row_stack((
            np.column_stack((
                np.array([1]),
                np.zeros((1, self.num_vars * self.num_lags))
            )),
            self.Φ_plus,
            np.column_stack((
                np.zeros((self.num_vars * (self.num_lags - 1), 1)),
                np.eye(self.num_vars * (self.num_lags - 1)),
                np.zeros((self.num_vars * (self.num_lags - 1), self.num_vars))
            ))
        ))

        # If no initial conditions were provided, construct the matrix of initial conditions
        if init_conditions is None:
            init_conditions = np.vstack(
                [
                    np.array([1]),
                    np.vstack(
                        [self.estimation_data[(start_period - 1) + (self.num_lags - l)] for l in range(self.num_lags)]
                    ).flatten().reshape(-1, 1)
                ]
            )

        # Compute the forecast
        y_th = forecast_selection_mat @ np.linalg.matrix_power(forecast_aux_mat, horizon) @ init_conditions

        # Compute the estimator of E[Z_t Z_t']
        Λ = self.Z @ self.Z.T / self.num_obs

        # Compute the selection matrix for computing the MA coefficient matrices
        MA_selection_mat = np.column_stack((
            np.eye(self.num_vars),
            np.zeros((self.num_vars, self.num_vars * (self.num_lags - 1)))
        ))

        # Compute the h step ahead MSE of the endogenous variables
        yh_MSE = np.zeros((self.num_vars, self.num_vars))
        for i in range(horizon):
            # Compute the ith MA coefficient matrix
            Δ_i = MA_selection_mat @ np.linalg.matrix_power(self.Γ, i) @ MA_selection_mat.T

            # Update yh_MSE
            yh_MSE += Δ_i @ self.Σ_ε @ Δ_i.T

        # Compute the asymptotic covariance of the forecast error
        fe_asy_cov = np.zeros((self.num_vars, self.num_vars))
        for i in range(horizon):
            for j in range(horizon):
                trace_term = np.trace(
                    np.linalg.matrix_power(forecast_aux_mat.T, horizon - 1 - i) @ np.linalg.inv(Λ) @ np.linalg.matrix_power(forecast_aux_mat, horizon - 1 - j) @ Λ
                )

                Δ_i = MA_selection_mat @ np.linalg.matrix_power(self.Γ, i) @ MA_selection_mat.T
                Δ_j = MA_selection_mat @ np.linalg.matrix_power(self.Γ, j) @ MA_selection_mat.T

                fe_asy_cov += trace_term * Δ_i @ self.Σ_ε @ Δ_j.T

        est_forecast_MSE = yh_MSE + fe_asy_cov / self.num_obs

        return y_th, est_forecast_MSE