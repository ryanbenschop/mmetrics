import numpy as np
from varmodel import VARModel
from typing import Callable, Tuple

"""
The following code defines a class for conducting Bayesian estimation of reduced form VAR models. The model can be written in the form:
y_t = μ + \sum_{l = 1}^p Φ_l y_{t - l} + ε_t
where y_t consists of a set of k dependent variables, μ is a constant in R^k, Φ_1, ..., Φ_p are k by k matrices, and ε_t ~ N(0, Σ) consists of k reduced form shocks.

The BVARModel class consists of a 2D array of observations on the dependent variables, and draws samples from the posterior distributions of β := vec(μ, Φ_1, ..., Φ_p) and
Σ. The available prior distributions include the Minnesota prior, natural conjugate prior, and the independent Gaussian-inverse Weishart prior.
"""

class BVARModel:
    def __init__(self):
        self.data = None                # Data for estimation
        
        self.num_lags = None            # Number of lags in the model
        self.num_obs = None             # Number of observations
        self.num_vars = None            # Number of endogenous variables
      
        self.var = None                 # VAR model used for OLS estimation
        self.resid_covariance = None    # OLS residual covariance estimate

        self.num_samples = None         # Number of samples of parameters to draw
        self.coeff_samples = None       # Samples of the coefficients drawn from the posterior
    
    #----------------------- OLS estimation of the model
    def estimate_ols(self, data: np.ndarray, num_lags: int, const: bool = True, cov_type: str = "db"):
        """
        Estimates the VAR model using OLS.

        Args:
            data (np.ndarray): A 2D numpy array where the rows represent time periods and the columns represent endogenous variables.
            num_lags (int): The number of lags to include in the VAR model.
            const (bool): = True if a constant should be included in the model.
            cov_type (str): Determines which estimator of the residual covariance to use. If set to "db", the debiased version of the ML estimator is used.
                Otherwise, the standard ML estimator is used.
        """

        # Create and estimate the VAR model
        var = VARModel()
        var.estimate_ols(data, num_lags, const, cov_type)

        # Update the class attributes
        self.data = data
        self.var = var
        self.num_lags = num_lags
        self.num_obs, self.num_vars = np.shape(data)
        self.resid_covariance = var.Σ_ε

    #----------------------- Prior definition and posterior sampling for Minnesota priors
    def construct_Minnesota_priors(
            self,
            const_std: float,
            self_lag_std_scale: float,
            diff_lag_rel_std: float,
            β_prior_type: str = "rw",
            β_prior_mean: np.ndarray = None
    ):
        """
        Construct the prior mean and covariance matrix of β based on the Minnesota prior. This assumes:
        β ~ N(prior_mean, prior_covariance)
        where the prior mean is typically defined to describe each endogenous variable as a univariate random walk, or to be zero, describe each endogenous variable
        as a white noise process, and the prior covariance is a diagonal matrix, constructed based on the hyperparameters and the OLS estimate of the error
        covariance.

        Args:
            const_std (float): Hyperparameter determining the prior standard deviation of the constant term in the VAR model.
            self_lag_std_scale (float): Hyperparameter determining the scale of the standard deviation of the coefficients on self lags of variables.
            diff_lag_rel_std (float): Hyperparameter determining the relative standard deviation of the coefficients on variables.
            β_prior_type (str): = "rw" if the prior mean corresponds with a univariate random walk for each variable, or = "wn" if the prior mean corresponds with
                a white noise process for each variable. Not used if β_prior_mean is specified manually.
            β_prior_mean (np.ndarray): Manually specifies the prior mean.
        """

        # Check that the OLS estimates have been computed
        assert self.resid_covariance is not None, """
            Unable to construct Minnesota prior. BVARModel.estimate_ols should be called before BVAR.construct_Minnesota_priors.
        """

        # Check that the hyperparameters are valid
        assert const_std >= 0, "const_std must be nonnegative."
        assert self_lag_std_scale >= 0, "self_lag_std_scale must be nonnegative."
        assert 0 <= diff_lag_rel_std <= 1, "diff_lag_rel_std must be between 0 and 1."
        assert β_prior_type in {"rw", "wn"}, "β_prior_type must be 'rw' or 'wn'."
        
        # Get the number of lags and endogenous variables in the model
        num_lags, num_vars = self.num_lags, self.num_vars

        # Get the estimate of the error covariance
        resid_covariance = self.resid_covariance

        # Total number of parameters in β
        num_params_β = num_vars * (1 + num_lags * num_vars)

        # If the prior mean is specified, check that it has the correct size
        if β_prior_mean is not None:
            assert β_prior_mean.shape == (num_params_β, 1), f"β_prior_mean must have shape ({num_params_β}, 1)."

        # Initialise the prior mean if it is not specified
        if not β_prior_mean:
            β_prior_mean = np.zeros((num_params_β, 1))
        
        # Initialise the prior variance
        β_prior_covariance = np.zeros((num_params_β, num_params_β))

        # Populate the prior mean and variance
        curr_lag = 0        # Track the current lag in the following loop
        curr_var = 0        # Track the current variable in the following loop
        curr_eqn = 1        # Track the current equation in the following loop

        for i in range(num_params_β):
            # Update the prior variance coreresponding with the constant term
            if curr_var == 0:
                β_prior_covariance[i][i] = const_std**2

            # Update the prior variance corresponding with the lags of a variable in its own equation
            if curr_eqn == curr_var:
                β_prior_covariance[i][i] = (self_lag_std_scale / curr_lag)**2

            # Update the prior variance corresponding with the lags of a variable in the other variables' equations
            if curr_eqn != curr_var and curr_lag != 0:
                β_prior_covariance[i][i] = (self_lag_std_scale * diff_lag_rel_std / curr_lag)**2 * (resid_covariance[curr_eqn - 1][curr_eqn - 1] / resid_covariance[curr_var - 1][curr_var - 1])

            # Update the prior mean if prior mean type is rw and the prior mean was not specified
            if β_prior_type == "rw" and curr_eqn == curr_var and curr_lag == 1:
                β_prior_mean[i] = 1

            # Update the current lag, variable, and equation appropriately
            if curr_lag == 0:
                curr_lag += 1
            elif (i - curr_eqn + 1) % num_vars == 0:
                curr_lag += 1

                if curr_lag > num_lags:
                    curr_lag = 0
            
            curr_var = 1 if curr_var == num_vars else curr_var + 1

            if (i + 1) % (num_vars * num_lags + 1) == 0:
                curr_eqn += 1
                curr_var = 0

        return β_prior_mean, β_prior_covariance

    def construct_Minnesota_posterior(
            self,
            data: np.ndarray,
            num_lags: int,
            const_std: float,
            self_lag_std_scale: float,
            diff_lag_rel_std: float,
            β_prior_type: str,
            β_prior_mean: np.ndarray = None,
            ols_const: bool = True,
            ols_cov_type: str = "db"
    ):
        """
        Construct the posterior mean and covariance matrix of β based on the Minnesota prior, using the standard analytical characterisation.

        Args:
            data (np.ndarray): A 2D numpy array where the rows represent time periods and the columns represent endogenous variables.
            num_lags (int): The number of lags to include in the VAR model.
            const_std (float): Hyperparameter determining the prior standard deviation of the constant term in the VAR model.
            self_lag_std_scale (float): Hyperparameter determining the scale of the standard deviation of the coefficients on self lags of variables.
            diff_lag_rel_std (float): Hyperparameter determining the relative standard deviation of the coefficients on variables.
            β_prior_type (str): = "rw" if the prior mean corresponds with a univariate random walk for each variable, or = "wn" if the prior mean corresponds with
                a white noise process for each variable. Not used if β_prior_mean is specified manually.
            β_prior_mean (np.ndarray): Manually specifies the prior mean.
            ols_const (bool): = True if a constant should be included in the model when computing the OLS estimate of the error covariance.
            ols_cov_type (str): Determines which estimator of the residual covariance to use when computing the OLS estimate of the error covariance. If set to
                "db", the debiased version of the ML estimator is used. Otherwise, the standard ML estimator is used.
        """

        # Check that the hyperparameters are valid
        assert const_std >= 0, "const_std must be nonnegative."
        assert self_lag_std_scale >= 0, "self_lag_std_scale must be nonnegative."
        assert 0 <= diff_lag_rel_std <= 1, "diff_lag_rel_std must be between 0 and 1."
        assert β_prior_type in {"rw", "wn"}, "β_prior_type must be 'rw' or 'wn'."

        # Compute the OLS estimate of the VAR model
        self.estimate_ols(data, num_lags, ols_const, ols_cov_type)
        self.var.estimate_ols(data, num_lags, ols_const, ols_cov_type)

        # Construct the regressand matrix
        y = np.row_stack((
            [self.data[num_lags:, i].reshape((-1, 1)) for i in range(self.num_vars)]
        ))

        # Construct the regressor matrix
        X = self.var.Z.T

        # Compute the prior mean and covariance
        β_prior_covariance = None

        if β_prior_mean:
            β_prior_mean, β_prior_covariance = self.construct_Minnesota_priors(
                const_std, self_lag_std_scale, diff_lag_rel_std, β_prior_type, β_prior_mean
            )
        else:
            β_prior_mean, β_prior_covariance = self.construct_Minnesota_priors(
                const_std, self_lag_std_scale, diff_lag_rel_std, β_prior_type
            )

        # Compute the posterior mean and covariance
        β_posterior_covariance = np.linalg.inv(
            np.linalg.inv(β_prior_covariance) + np.kron(np.linalg.inv(self.resid_covariance), X.T @ X)
        )

        β_posterior_mean = β_posterior_covariance @ (np.linalg.inv(β_prior_covariance) @ β_prior_mean + (np.kron(np.linalg.inv(self.resid_covariance), X)).T @ y)

        return β_posterior_mean, β_posterior_covariance
    
    def sample_Minnesota_posterior(
            self,
            data: np.ndarray,
            num_lags: int,
            const_std: float,
            self_lag_std_scale: float,
            diff_lag_rel_std: float,
            β_prior_type: str,
            num_samples: int,
            β_prior_mean: np.ndarray = None,
            ols_const: bool = True,
            ols_cov_type: str = "db"
    ):
        """
        Draw parameter samples from the posterior distribution of β based on the Minnesota prior.

        Args:
            data (np.ndarray): A 2D numpy array where the rows represent time periods and the columns represent endogenous variables.
            num_lags (int): The number of lags to include in the VAR model.
            const_std (float): Hyperparameter determining the prior standard deviation of the constant term in the VAR model.
            self_lag_std_scale (float): Hyperparameter determining the scale of the standard deviation of the coefficients on self lags of variables.
            diff_lag_rel_std (float): Hyperparameter determining the relative standard deviation of the coefficients on variables.
            β_prior_type (str): = "rw" if the prior mean corresponds with a univariate random walk for each variable, or = "wn" if the prior mean corresponds with
                a white noise process for each variable. Not used if β_prior_mean is specified manually.
            β_prior_mean (np.ndarray): Manually specifies the prior mean.
            ols_const (bool): = True if a constant should be included in the model when computing the OLS estimate of the error covariance.
            ols_cov_type (str): Determines which estimator of the residual covariance to use when computing the OLS estimate of the error covariance. If set to
                "db", the debiased version of the ML estimator is used. Otherwise, the standard ML estimator is used.
        """

        assert num_samples > 0, "num_samples must be a positive integer."
        assert num_lags >= 0, "num_lags must be a nonnegative integer."

        # Compute the posterior mean and covariance
        β_posterior_mean, β_posterior_covariance = self.construct_Minnesota_posterior(
            data, num_lags, const_std, self_lag_std_scale, diff_lag_rel_std, β_prior_type, β_prior_mean, ols_const, ols_cov_type
        )

        # Draw samples from the posterior distribution
        β_samples = np.random.multivariate_normal(β_posterior_mean.flatten(), β_posterior_covariance, num_samples)

        self.num_samples = num_samples
        self.coeff_samples = β_samples

    #----------------------- Impulse response functions
    def compute_reduced_form_IRFs(
            self,
            Γ: np.ndarray,
            horizon: int,
            num_lags: int,
            num_vars: int,
            J: np.ndarray = None
    ):
        """
        Compute the reduced form impulse response function matrix for a given horizon.

        Args:
            Γ (np.ndarray): Transition matrix that represents the VAR structure.
            horizon (int): The forecast horizon for the IRFs.
            num_lags (int): Number of lags in the VAR model.
            num_vars (int): Number of endogenous variables in the model.
            J (np.ndarray, optional): Selection matrix that isolates contemporaneous impacts. 
                                      Defaults to None, in which case a standard selection matrix is created.

        Returns:
            np.ndarray: Reduced-form IRF matrix at the specified horizon.
        """

        # Initialize the selection matrix if none is provided
        if J is None:
            J = np.column_stack((
                np.eye(num_vars),
                np.zeros((num_vars, num_vars * (num_lags - 1)))
            ))

        # Compute the reduced-form IRF at the specified horizon
        Ξ_h = J @ np.linalg.matrix_power(Γ, horizon) @ J.T

        return Ξ_h
    
    def compute_SIRFs(
            self,
            Γ: np.ndarray,
            Θ_0: np.ndarray,
            horizon: int,
            num_lags: int,
            num_vars: int,
            J: np.ndarray = None
    ):
        """
        Compute the structural impulse response function matrix for a given horizon.

        Args:
            Γ (np.ndarray): Transition matrix representing the VAR structure.
            Θ_0 (np.ndarray): Structural impact matrix, mapping structural shocks to reduced-form shocks.
            horizon (int): Forecast horizon for the SIRFs.
            num_lags (int): Number of lags in the VAR model.
            num_vars (int): Number of endogenous variables in the model.
            J (np.ndarray, optional): Selection matrix that isolates contemporaneous impacts. 
                                      Defaults to None, in which case a standard selection matrix is created.

        Returns:
            np.ndarray: Structural IRF matrix at the specified horizon.
        """

        # Compute the reduced-form IRF matrix at the specified horizon
        Ξ_h = self.compute_reduced_form_IRFs(Γ, horizon, num_lags, num_vars, J)

        # Transform the reduced-form IRFs into structural IRFs by applying the impact matrix Θ_0
        Θ_h = Ξ_h @ Θ_0

        return Θ_h

    def sample_SIRFs(
            self,
            Θ_0_identifier: Callable[[np.ndarray, np.ndarray], np.ndarray],
            max_horizon: int
    ):      
        """
        Compute samples of structural impulse response functions across posterior samples of the VAR coefficients and residual covariance.

        Args:
            Θ_0_identifier (Callable): Function that computes the structural impact matrix Θ_0 from a given set of VAR coefficients and residual covariance.
            max_horizon (int): Maximum forecast horizon for the impulse responses.

        Returns:
            list: A list of SIRF samples for each posterior sample, where each SIRF sample is a list of IRF matrices for each horizon up to max_horizon.
        """

        # Reshape coefficient samples. This undoes the vectorisation applied when drawing the coefficient samples
        Φ_plus_samples = self.coeff_samples.reshape((self.num_samples, self.num_vars, 1 + self.num_lags * self.num_vars))

        # Initialize list to store SIRF samples for each posterior draw
        sirf_samples = []

        # Loop over each sample of the VAR coefficients
        for Φ_plus in Φ_plus_samples:
            # Construct the transition matrix Γ from the VAR coefficients. This is the matrix associated with the companion form of the VAR(p)
            Γ = np.row_stack((
                Φ_plus[:, 1:],
                np.column_stack((np.eye(self.num_vars * (self.num_lags - 1)), np.zeros((self.num_vars * (self.num_lags - 1), self.num_vars))))
            ))

            # Define the selection matrix J
            J = np.column_stack((
                np.eye(self.num_vars),
                np.zeros((self.num_vars, self.num_vars * (self.num_lags - 1)))
            ))

            # Compute the structural impact matrix Θ_0 using the identifier function
            Θ_0 = Θ_0_identifier(Φ_plus, self.resid_covariance)

            # Compute SIRFs at the current parameter draw
            sirfs = [
                self.compute_SIRFs(Γ, Θ_0, h, self.num_lags, self.num_vars, J)
                for h in range(max_horizon + 1)
            ]

            # Store SIRFs for the current parameter draw
            sirf_samples.append(sirfs)

        return sirf_samples
    
    #----------------------- Forecasting
    def compute_point_forecast(self, horizon, Φ_plus, Σ_ε):
        var_model = VARModel()
        var_model.estimation_data = self.data
        var_model.num_obs = self.num_obs - self.num_lags
        _, Z = var_model.extract_regression_matrices(self.data, self.num_lags, True)
        var_model.Z = Z
        var_model.set_parameters(Φ_plus, Σ_ε)

        # y_th, _ = var_model.forecast(horizon)
        y_th = var_model.forecast(horizon)

        return y_th
    
    def compute_forecasts(self, max_horizon, Φ_plus, Σ_ε):
        forecasts = [self.compute_point_forecast(h, Φ_plus, Σ_ε) for h in range(1, max_horizon + 1)]

        return forecasts
    
    def compute_forecast_samples(self, max_horizon):
        forecast_samples = []

        Φ_plus_samples = self.coeff_samples.reshape((self.num_samples, self.num_vars, 1 + self.num_lags * self.num_vars))
        
        for Φ_plus in Φ_plus_samples:
            forecast_samples.append(self.compute_forecasts(max_horizon, Φ_plus, self.resid_covariance))

        return forecast_samples
