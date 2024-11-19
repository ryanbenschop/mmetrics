# MMetrics

## Overview
The MMetrics library provides an general Python code for the implementation of standard methods in macroeconometric modelling, with an emphasis on VAR based modelling. This library was
primarily developed as my own preparation for the Macroeconometrics course at the University of Oxford, as part of the MPhil in Economics, as presented in Hilary Term 2024.

## Features

### 1. Reduced-Form Vector Autoregressive Analysis
- Estimation of reduced form VAR models using ordinary least squares with heteroskedasticity robust standard errors.
- Computation of pointwise impulse response functions, and confidence intervals based on the residual based recursive and wild bootstrap.
- Computation of point forecast, and mean standard errors used for interval forecating.
- Implementation of information criteria-based lag order selection procedures.
- Implementation of residual-based model diagnostic techniques.

### 2. Structural Vector Autoregressive Analysis
- Least squares and Bayesian estimation of structural vector autoregressive models based on recursive identification schemes.
  - Alternative identification schemes may be specified in the form of a function of the error covariance.
- Computation of pointwise structural impulse response functions, and bootstrap and Bayesian confidence intervals.
- Computation of the structural forecast error decomposition.
- Structural forecast scenario analysis.

### 3. Bayesian Vector Autoregressive Analysis
- Direct posterior sampling of the autorgressive coefficient matrices based on the Minnesota prior.
- Computation of posterior predictive distribution forecasts.
