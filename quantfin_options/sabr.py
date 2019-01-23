# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
SABR model related formulas.

The Stochastic Alpha Beta Rho model is the classic SABR model.
For references, see "Managing Smile Risk" by Hagan, Kumar, Lesniewski and Woodward (2002).

"""

import numpy as np
from scipy.optimize import newton, minimize, Bounds
from scipy.stats import linregress

from .black_scholes import convert_impl_vol

DTYPE = np.float


def volatility(fwd, strike, tau, alpha, beta, rho, nu, model='l', boundary=0.0):
    """
    Returns the SABR model implied volatility to be used in Black-Scholes formula.

    Parameters
    ----------
    fwd : float or ndarray of float
        Forwards.
    strike : float or ndarray of float
        Strikes.
    tau : float or ndarray of float
        Time to expiry, normally in years assuming that volatilities are "per year equivalent".
    alpha : float or ndarray of float
        The :math:`\alpha` parameter, mostly influencing overall option prices, regardless of skew behaviour.
    beta : float or ndarray of float
        The :math:`\beta` parameter, to define the behaviour of the backbone.
    rho : float or ndarray of float
        The Rho or correlation parameter, for correlations between asset price and volatility.
    nu : float or ndarray of float
        The Nu or stochastic volatility parameter.
    model : {'l', 'n', 'bn'}, optional
        Volatility model type: lognormal, normal, bounded normal.
        l = lognormal (black) [default] | n = normal | bn = bounded normal
    boundary : float, optional
        Boundary for the bounded normal model. If model = 'bn', bound needs to be specified.

    Returns
    -------
    sigma : float or ndarray of float
        The volatility :math:`\sigma` to use in the standard Black-Scholes formula (assuming the same model).

    """

    # first calculate all standard SABR related things, then check for volatility conversion if needed
    # note that I make a check that logfoverk is greater than epsilon
    logfoverk = np.log(fwd / strike)
    fk1_b = np.power(fwd * strike, 1 - beta)
    fk1_b2 = np.sqrt(fk1_b)
    omb = 1 - beta
    zeta = nu / alpha * fk1_b2 * logfoverk
    xi = np.log((np.sqrt(1.0 - 2.0 * rho * zeta + zeta**2) + zeta - rho) / (1.0 - rho))
    msk = np.abs(logfoverk) > np.finfo(DTYPE).resolution
    zetaoverxi = np.divide(zeta, xi, out=np.ones_like(zeta), where=msk)
    if zetaoverxi.ndim == 0:
        zetaoverxi = np.asscalar(zetaoverxi)

    if model == "l":
        sigma = alpha / fk1_b2 / (1.0 + 1.0 / 24.0 * omb**2 * logfoverk**2
                                + 1.0 / 1920.0 * omb**4 * logfoverk**4) * zetaoverxi \
              * (1.0 + (1.0 / 24.0 * omb**2 * alpha**2 / fk1_b + 0.25 * rho * beta * nu * alpha / fk1_b2
                        + (2.0 - 3.0 * rho**2) / 24.0 * nu**2) * tau)
        return sigma
    elif model == "n" or model == "bn":
        sigma = alpha * np.power(fwd * strike, beta / 2.0) \
              * (1.0 + 1.0 / 24.0 * logfoverk**2 + 1.0 / 1920.0 * logfoverk**4) \
              / (1.0 + 1.0 / 24.0 * omb**2 * logfoverk**2 + 1.0 / 1920.0 * omb**4 * logfoverk**4) \
              * zetaoverxi \
              * (1.0 + (-1.0 / 24.0 * beta * (2.0 - beta) * alpha**2 / fk1_b
                        + 0.25 * rho * beta * nu * alpha / fk1_b2
                        + (2.0 - 3.0 * rho**2) / 24.0 * nu**2) * tau)
        if model == "n":
            return sigma
        else:
            return convert_impl_vol(fwd, strike, tau, sigma, 'n', 'bn', 0.0, boundary)
    else:
        raise NotImplementedError("Model not implemented in this function.")


def alpha(fwd, tau, sigma, beta=0.0, rho=0.0, nu=0.0, model='l', boundary=0.0):
    """
    Calculates the implied SABR :math:`\alpha` based on the provided :math:`\sigma` and the other SABR parameters
    for an at the money option.
    In this formula, the assumption is used that fwd == strike and that the options are therefore at the money.

    Parameters
    ----------
    fwd : float or ndarray of float
        Forwards and strikes.
    tau : float or ndarray of float
        Time to expiry, normally in years assuming that volatilities are "per year equivalent".
    sigma : float or ndarray of float
        The volatility backsolved from a black-scholes pricer that are used to calculate alpha.
    beta : float or ndarray of float, optional.
        The :math:`\beta` parameter, to define the behaviour of the backbone.
        Defaults to beta = 0.0.
    rho : float or ndarray of float, optional.
        The Rho or correlation parameter, for correlations between asset price and volatility.
        Defaults to rho = 0.0.
    nu : float or ndarray of float, optional.
        The Nu or stochastic volatility parameter.
        Defaults to nu = 0.0.
    model : {'l', 'n', 'bn'}, optional
        Volatility model type: lognormal, normal, bounded normal.
        l = lognormal (black) [default] | n = normal | bn = bounded normal
    boundary : float, optional
        Boundary for the bounded normal model. If model = 'bn', bound needs to be specified.

    Returns
    -------
    alpha : float or ndarray of float
        The :math:`\alpha` parameter for the SABR implementation.

    """

    fwd_mod = np.atleast_1d(fwd)
    sigma_mod = np.atleast_1d(sigma)
    tau_mod = np.atleast_1d(tau)
    beta_mod = np.atleast_1d(beta)
    rho_mod = np.atleast_1d(rho)
    nu_mod = np.atleast_1d(nu)
    if model == 'l':
        al_guess_mod = np.atleast_1d(sigma * fwd**(1.0 - beta))
    elif model == 'n' or model == 'bn':
        al_guess_mod = np.atleast_1d(sigma / fwd**beta)
    else:
        raise NotImplementedError("Model not implemented in this function.")

    def func(x, pos):
        return sigma_mod[pos] - volatility(fwd_mod[pos], fwd_mod[pos], tau_mod[pos], x, beta_mod[pos], rho_mod[pos],
                                           nu_mod[pos], model, boundary)

    al = np.empty(fwd_mod.shape, dtype=DTYPE)
    for i in range(al.size):
        al[i] = newton(func, al_guess_mod[i], args=(i,))
    if al.size == 1:
        al = np.asscalar(al)
    return al


def beta_estimate(fwd, sigma, model='l'):
    """
    Estimates the :math:`\beta` parameter based on a linear regression of simplified SABR formulas for at the money
    combinations of forwards and alpha parameters representing the volatility.

    Parameters
    ----------
    fwd : ndarray of float
        Forwards.
    sigma : ndarray of float
        The :math:`\sigma` parameter representing the at the money volatility to be used in Black-Scholes formulas.
    model : {'l', 'n'}
        Volatility model type: lognormal, normal.
        l = lognormal (black) [default] | n = normal

    Returns
    -------
    beta : float
        The regression-based estimate for the SABR :math:`\beta` parameter estimate.

    """

    # if fwd = strike the normal and lognormal formulas simplify substantially
    # and depend on time only on a small order. As a consequence, the relationship can be estimated via regression
    # on logarithms

    x = np.log(fwd)
    y = np.log(sigma)
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    if model == 'l':
        return 1 + slope
    elif model == 'n':
        return slope
    else:
        raise NotImplementedError("Model not implemented in this function.")


def calibrate(fwd, tau, atm_sigma, skew_k, skew_sigma, beta=None, skew_weights=None, skew_k_relative=False,
              skew_sigma_relative=False, fit_as_sequence=True, model='l', boundary=0.0):
    """
    Calibrates SABR parameters based on skew information. 
    The :math:`\beta` parameter can be fixed, in which case only :math:`\alpha`, :math:`\rho` and :math:`\nu`
    will be calculated.
    Skew information can be provided both in relative terms (vs. at the money forward and at the money volatility)
    and in absolute terms.
    In case of "temporal" or "spatial" sequences, the convergence can be accelerated using the boolean
    parameter fit_as_sequence, as the loop uses the prior step converged results as starting point for the
    iterative error-minimisation procedure.
    Parameter :math:`\rho` will be limited between -1 and 1, :math:`\alpha` and :math:`\nu` will be strictly positive
    while :math:`\beta` can be unbounded.

    Parameters
    ----------
    fwd : float or ndarray of float
        Forwards.
    tau : float or ndarray of float
        Time to expiry, normally in years assuming that volatilities are "per year equivalent".
    atm_sigma : float or ndarray of float
        Provide the quantification of the at the money volatility.
        An :math:`\alpha` guess will be calculated first and then refined.
    skew_k : ndarray of float
        Strikes (relative or absolute) for which implied volatility is provided.
    skew_sigma : ndarray of float
        Skew implied volatility at the given strikes.
    beta : float or ndarray of float
        SABR :math:`\beta` that is assumed to be known and fixed for calibration of the other parameters.
    skew_weights : ndarray of float, optional
        Use weights on the errors of the skew implied volatilities.
        Defaults to None, in which case all weights are assumed to be 1.
    skew_k_relative : bool, optional
        Use the skew strikes as relative to the forward or absolute.
        True = Strikes are relative | False = Strikes are absolute (default)
    skew_sigma_relative : bool, optional
        Use the skew volatilities as relative to the at the money volatility or absolute.
        True = Skew volatilities are relative | False = Skew volatilities are absolute (default)
    fit_as_sequence : bool, optional
        Only relevant if fwd and related variables is ndarray: if True (default), the optimisation algorithm
        shall consider the result of the prior step optimisation as starting point for the new step, for example when
        fitting multiple smiles in a time sequence, or when fitting smiles that are related (e.g. expiry).
        True = fit as sequence | False = ignore any relationship that may speed up convergence.
    model : {'l', 'n', 'bn'}, optional
        Volatility model type: lognormal, normal, bounded normal.
        l = lognormal (black) [default] | n = normal | bn = bounded normal
    boundary : float, optional
        Boundary for the bounded normal model. If model = 'bn', bound needs to be specified.

    Returns
    -------
    alpha : float or ndarray of float
        The SABR :math:`\alpha` parameter.
    rho : float or ndarray of float
        The SABR :math:`\rho` parameter.
    nu : float or ndarray of float
        The SABR :math:`\nu` parameter.
    beta : float or ndarray of float
        The SABR :math:`\beta` parameter (may be the same as the one in input).
    err : ndarray of float
        The calibration errors.

    """

    fwd_mod = np.atleast_1d(fwd)
    tau_mod = np.atleast_1d(tau)
    atm_sigma_mod = np.atleast_1d(atm_sigma)
    skew_k_mod = np.atleast_2d(skew_k)
    skew_sigma_mod = np.atleast_2d(skew_sigma)
    if beta is not None:
        beta_mod = np.atleast_1d(beta)
    if skew_weights is not None:
        skew_weights_mod = np.atleast_1d(skew_weights)
    else:
        skew_weights_mod = np.ones(shape=(skew_k_mod.shape[1],), dtype=DTYPE)
    if skew_k_relative:
        skew_k_mod = fwd_mod[:, np.newaxis] + skew_k_mod
    if skew_sigma_relative:
        skew_sigma_mod = atm_sigma_mod[:, np.newaxis] + skew_sigma_mod

    # Define the functions to minimize
    def alphabetarhonu(x, pos):
        # x contains alpha, beta, rho, nu in this order, pos is the position
        # that we are in the loop
        v = volatility(fwd_mod[pos], skew_k_mod[pos,:], tau_mod[pos],
                       x[0], x[1], x[2], x[3], model=model, boundary=boundary)
        vatm = volatility(fwd_mod[pos], fwd_mod[pos], tau_mod[pos],
                          x[0], x[1], x[2], x[3], model=model, boundary=boundary)
        e = np.dot((skew_sigma_mod[pos, :] - v)**2, skew_weights_mod) + (atm_sigma_mod[pos] - vatm)**2
        return e

    def alpharhonu(x, pos):
        # x contains alpha, rho, nu in this order, pos is the position
        # that we are in the loop
        v = volatility(fwd_mod[pos], skew_k_mod[pos, :], tau_mod[pos],
                       x[0], beta_mod[pos], x[1], x[2], model=model, boundary=boundary)
        vatm = volatility(fwd_mod[pos], fwd_mod[pos], tau_mod[pos],
                          x[0], beta_mod[pos], x[1], x[2], model=model, boundary=boundary)
        e = np.dot((skew_sigma_mod[pos, :] - v) ** 2, skew_weights_mod) + (atm_sigma_mod[pos] - vatm)**2
        return e

    alpha_out = np.empty(shape=fwd_mod.shape, dtype=DTYPE)
    rho_out = np.empty(shape=fwd_mod.shape, dtype=DTYPE)
    nu_out = np.empty(shape=fwd_mod.shape, dtype=DTYPE)
    err_out = np.empty(shape=fwd_mod.shape, dtype=DTYPE)
    if beta is None:
        # Beta needs to be also part of the optimisation
        beta_out = np.empty(shape=fwd_mod.shape, dtype=DTYPE)
        x0 = np.empty(shape=(fwd_mod.shape[0], 4), dtype=DTYPE)
        beta_idx = 1
        rho_idx = 2
        nu_idx = 3
        x0[:, beta_idx] = 0.5  # i assign beta here but note that it will be checked later as well
        x0[:, rho_idx] = 0.0  # rho
        x0[:, nu_idx] = 0.5  # nu
        x0[:, 0] = alpha(fwd_mod, tau_mod, atm_sigma_mod,
                         beta=x0[:, beta_idx], rho=x0[:, rho_idx], nu=x0[:, nu_idx],
                         model=model, boundary=boundary)
        func = alphabetarhonu
        bnd = Bounds([0.0, -np.Inf, -1.0, 0.0], [np.Inf, np.Inf, 1.0, np.Inf])
    else:
        # Beta is provided externally
        beta_out = beta_mod
        x0 = np.empty(shape=(fwd_mod.shape[0], 3), dtype=DTYPE)
        rho_idx = 1
        nu_idx = 2
        beta_idx = None
        x0[:, rho_idx] = 0.0  # rho
        x0[:, nu_idx] = 0.5  # nu
        x0[:, 0] = alpha(fwd_mod, tau_mod, atm_sigma_mod,
                         beta=beta_mod, rho=x0[:, rho_idx], nu=x0[:, nu_idx],
                         model=model, boundary=boundary)
        func = alpharhonu
        bnd = Bounds([0.0, -1.0, 0.0], [np.Inf, 1.0, np.Inf])

    for i in range(fwd_mod.size):
        x0_start = x0[i, :]
        if fit_as_sequence and i > 0:
            x0_start[0] = alpha_out[i - 1]
            x0_start[rho_idx] = rho_out[i - 1]
            x0_start[nu_idx] = nu_out[i - 1]
            if beta_idx is not None:
                x0_start[beta_idx] = beta_out[i - 1]
        opt_res = minimize(func, x0_start, args=(i,), method='L-BFGS-B', bounds=bnd)
        if opt_res.success:
            alpha_out[i] = opt_res.x[0]
            rho_out[i] = opt_res.x[rho_idx]
            nu_out[i] = opt_res.x[nu_idx]
            if beta is None:
                beta_out[i] = opt_res.x[beta_idx]
            err_out[i] = opt_res.fun
        else:
            raise RuntimeError(opt_res.message)

    if alpha_out.size == 1:
        alpha_out = np.asscalar(alpha_out)
        beta_out = np.asscalar(beta_out)
        rho_out = np.asscalar(rho_out)
        nu_out = np.asscalar(nu_out)
        err_out = np.asscalar(err_out)

    return alpha_out, beta_out, rho_out, nu_out, err_out

