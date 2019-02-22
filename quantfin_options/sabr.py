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

from .black_scholes import convert_impl_vol, fwd_value

DTYPE = np.float

def fwd_price(fwd, strike, tau, vol, beta, rho, nu, vol_type='alpha', opt_type='c', 
              model='l', boundary=0.0):
    """
    Returns the forward option price based on SABR model parameters.
    
    Parameters
    ----------
    fwd : float or ndarray of float
        Forwards.
    strike : float or ndarray of float
        Strikes.
    tau : float or ndarray of float
        Time to expiry, normally in years assuming that volatilities are "per year equivalent".
    vol : float or ndarray of float
        The at the money vol parameter, mostly influencing overall option prices, regardless of 
        skew behaviour. This can be the SABR alpha parameter directly, or the at the money 
        volatility, from which the alpha parameter will be derived.
    beta : float or ndarray of float
        The beta parameter, to define the behaviour of the backbone.
    rho : float or ndarray of float
        The Rho or correlation parameter, for correlations between asset price and volatility.
    nu : float or ndarray of float
        The Nu or stochastic volatility parameter.
    vol_type : {'alpha','atmvol'}, optional, default: alpha
        The type of the vol parameter.
        alpha = SABR alpha | atmvol = at the money implied vol according to model.
    opt_type : str or ndarray of str, optional
        Option type to price.
        c = call [default] | p = put | s = straddle
    model : {'l', 'n', 'bn'}, optional, default: l
        Volatility model type: lognormal, normal or bounded normal.
        l = lognormal (black) | n = normal | bn = bounded normal
    boundary : float, optional
        Boundary for the bounded normal model. If model = 'bn', bound cannot be None and needs to be specified.

    Returns
    -------
    price : float or ndarray of float
        The forward price of the option with the given SABR parameters.
    
    """
    if vol_type == 'alpha':
        al = vol
    else:
        al = alpha(fwd, tau, vol, beta=beta, rho=rho, nu=nu, model=model, boundary=boundary)
    
    sabr_vol = volatility(fwd, strike, tau, al, beta, rho, nu, model=model, boundary=boundary)
    price = fwd_value(fwd, strike, tau, sabr_vol, opt_type=opt_type, model=model, boundary=boundary)
    return price
    

def fwd_risks(fwd, strike, tau, alpha, beta, rho, nu, opt_type='c', model='l', boundary=0.0):
    raise NotImplementedError()

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
        The alpha parameter, mostly influencing overall option prices, regardless of skew behaviour.
    beta : float or ndarray of float
        The beta parameter, to define the behaviour of the backbone.
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
        The volatility sigma to use in the standard Black-Scholes formula (assuming the same model).

    """

    # first calculate all standard SABR related things, then check for volatility conversion if needed
    # note that I make a check that logfoverk is greater than epsilon
    logfoverk = np.log(fwd / strike)
    fk1_b = np.power(fwd * strike, 1 - beta)
    fk1_b2 = np.sqrt(fk1_b)
    omb = 1 - beta
    zeta = nu / alpha * fk1_b2 * logfoverk
    xi = np.log((np.sqrt(1.0 - 2.0 * rho * zeta + zeta**2) + zeta - rho) / (1.0 - rho))
    msk = np.abs(xi) > 1e-6     #put boundary on where xi could get too close to 0
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


def alpha(fwd, tau, sigma, beta=1.0, rho=0.0, nu=0.0, alpha_guess=None, model='l', boundary=0.0):
    """
    Calculates the implied SABR alpha based on the provided sigma and the other SABR parameters
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
    beta : float or ndarray of float, optional, default 1.0.
        The beta parameter, to define the behaviour of the backbone.
    rho : float or ndarray of float, optional, default 0.0.
        The Rho or correlation parameter, for correlations between asset price and volatility.
    nu : float or ndarray of float, optional, default 0.0.
        The Nu or stochastic volatility parameter.
    alpha_guess : float or ndarray of float, optional, default None.
        The initial guess for alpha to get faster convergence. 
    model : {'l', 'n', 'bn'}, optional
        Volatility model type: lognormal, normal, bounded normal.
        l = lognormal (black) [default] | n = normal | bn = bounded normal
    boundary : float, optional
        Boundary for the bounded normal model. If model = 'bn', bound needs to be specified.

    Returns
    -------
    alpha : float or ndarray of float
        The alpha parameter for the SABR implementation.

    """

    fwd_mod = np.atleast_1d(fwd)
    sigma_mod = np.atleast_1d(sigma)
    tau_mod = np.atleast_1d(tau)
    beta_mod = np.atleast_1d(beta)
    rho_mod = np.atleast_1d(rho)
    nu_mod = np.atleast_1d(nu)
    if alpha_guess is None:
        if model == 'l':
            al_guess_mod = np.atleast_1d(sigma * fwd**(1.0 - beta))
        elif model == 'n' or model == 'bn':
            al_guess_mod = np.atleast_1d(sigma / fwd**beta)
        else:
            raise NotImplementedError("Model not implemented in this function.")
    else:
        al_guess_mod = np.atleast_1d(alpha_guess)

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
    Estimates the beta parameter based on a linear regression of simplified SABR formulas for at the money
    combinations of forwards and alpha parameters representing the volatility.

    Parameters
    ----------
    fwd : ndarray of float
        Forwards.
    sigma : ndarray of float
        The sigma parameter representing the at the money volatility to be used in Black-Scholes formulas.
    model : {'l', 'n'}
        Volatility model type: lognormal, normal.
        l = lognormal (black) [default] | n = normal

    Returns
    -------
    beta : float
        The regression-based estimate for the SABR beta parameter estimate.

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
    The beta parameter can be fixed, in which case only alpha, rho and nu
    will be calculated.
    Skew information can be provided both in relative terms (vs. at the money forward and at the money volatility)
    and in absolute terms.
    In case of "temporal" or "spatial" sequences, the convergence can be accelerated using the boolean
    parameter fit_as_sequence, as the loop uses the prior step converged results as starting point for the
    iterative error-minimisation procedure.
    
    Parameter rho will be limited between -0.999 and 0.999 for numerical reasons, 
    alpha and nu greater than 0.01% 
    with nu limited to 10 (or 1000%) while beta will be limited between -5 and 5 (unless
    specified).

    Parameters
    ----------
    fwd : float or ndarray of float
        Forwards.
    tau : float or ndarray of float
        Time to expiry, normally in years assuming that volatilities are "per year equivalent".
    atm_sigma : float or ndarray of float
        Provide the quantification of the at the money volatility.
        An alpha guess will be calculated first and then refined.
    skew_k : ndarray of float
        Strikes (relative or absolute) for which implied volatility is provided.
    skew_sigma : ndarray of float
        Skew implied volatility at the given strikes.
    beta : float or ndarray of float
        SABR beta that is assumed to be known and fixed for calibration of the other parameters.
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
        The SABR alpha parameter.
    rho : float or ndarray of float
        The SABR rho parameter.
    nu : float or ndarray of float
        The SABR nu parameter.
    beta : float or ndarray of float
        The SABR beta parameter (may be the same as the one in input).
    err : ndarray of float
        The squared calibration errors.

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
    e_factor = 1000
    def alphabetarhonu(x, pos):
        # x contains alpha, beta, rho, nu in this order, pos is the position
        # that we are in the loop
        # and as volatility can be % or basis points (so numbers that are small), 
        # i multiply the error by the factor
        v = volatility(fwd_mod[pos], skew_k_mod[pos,:], tau_mod[pos],
                       x[0], x[1], x[2], x[3], model=model, boundary=boundary)
        vatm = volatility(fwd_mod[pos], fwd_mod[pos], tau_mod[pos],
                          x[0], x[1], x[2], x[3], model=model, boundary=boundary)
        e = np.dot((skew_sigma_mod[pos, :] - v)**2, skew_weights_mod) + (atm_sigma_mod[pos] - vatm)**2
        return e * e_factor**2

    def alpharhonu(x, pos):
        # x contains alpha, rho, nu in this order, pos is the position
        # that we are in the loop
        v = volatility(fwd_mod[pos], skew_k_mod[pos, :], tau_mod[pos],
                       x[0], beta_mod[pos], x[1], x[2], model=model, boundary=boundary)
        vatm = volatility(fwd_mod[pos], fwd_mod[pos], tau_mod[pos],
                          x[0], beta_mod[pos], x[1], x[2], model=model, boundary=boundary)
        e = np.dot((skew_sigma_mod[pos, :] - v) ** 2, skew_weights_mod) + (atm_sigma_mod[pos] - vatm)**2
        return e * e_factor**2

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
        x0[:, nu_idx] = 0.75  # nu 75%
        x0[:, 0] = alpha(fwd_mod, tau_mod, atm_sigma_mod,
                         beta=x0[:, beta_idx], rho=x0[:, rho_idx], nu=x0[:, nu_idx],
                         model=model, boundary=boundary)
        func = alphabetarhonu
        bnd = Bounds([0.0001, -5, -0.999, 0.0001], [np.Inf, 5, 0.999, 10])
    else:
        # Beta is provided externally
        beta_out = beta_mod
        x0 = np.empty(shape=(fwd_mod.shape[0], 3), dtype=DTYPE)
        rho_idx = 1
        nu_idx = 2
        beta_idx = None
        x0[:, rho_idx] = 0.0  # rho
        x0[:, nu_idx] = 0.75  # nu 75%
        x0[:, 0] = alpha(fwd_mod, tau_mod, atm_sigma_mod,
                         beta=beta_mod, rho=x0[:, rho_idx], nu=x0[:, nu_idx],
                         model=model, boundary=boundary)
        func = alpharhonu
        bnd = Bounds([0.0001, -0.999, 0.0001], [np.Inf, 0.999, 10])

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

    return alpha_out, beta_out, rho_out, nu_out, err_out / e_factor**2

