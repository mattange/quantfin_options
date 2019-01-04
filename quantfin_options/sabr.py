# -*- coding: utf-8 -*-
"""
SABR model related formulas.

The Stochastic Alpha Beta Rho model is the classic SABR model.
For references, see "Managing Smile Risk" by Hagan, Kumar, Lesniewski and Woodward (2002).

"""

import numpy as np
from .black_scholes import option_convert_impl_vol

DTYPE = np.float

def impl_vol(fwd, strike, tau, alpha, beta, rho, nu, model='l', boundary=0.0):
    """
    Returns the SABR model implied volatility to be used in Black-Scholes formula.

    Parameters
    ----------
    fwd : float or ndarray of float
        Forwards.
    strike : float or ndarray of float
        Strikes.
    tau : float or ndarray of float
        Times to expiry, normally in years assuming that volatilities are "per year equivalent".
    alpha : float or ndarray of float
        The :math:`\alpha` parameter, mostly influencing overall option prices, regardless of skew behaviour.
    beta : float or ndarray of float
        The :math:`\beta` parameter, to define the behaviour of the backbone.
    rho : float or ndarray of float
        The Rho or correlation parameter, for correlations between asset price and volatility.
    nu : float or ndarray of float
        The Nu or stochastic volatility parameter.
    model : {'l', 'n', 'bn'}, optional
        Volatility model type: lognormal, normal, boundaed normal.
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
            return option_convert_impl_vol(fwd, strike, tau, sigma, 'n', 'bn', 0.0, boundary)
    else:
        raise NotImplementedError("Model not implemented in this function.")

def alpha(fwd, tau, sigma, beta, rho, nu, model='l', boundary=0.0):
    """
    Calculates the implied SABR :math:`\alpha` based on the provided :math:`\sigma` and the other SABR parameters
    for an at the money option.
    In this formula, the assumption is used that fwd == strike and that the options are therefore at the money.

    Parameters
    ----------
    fwd : float or ndarray of float
        Forwards and strikes.
    tau : float or ndarray of float
        Times to expiry, normally in years assuming that volatilities are "per year equivalent".
    sigma : float or ndarray of float
        The volatilities backsolved from a black-scholes pricer that are used to calculate alpha.
    beta : float or ndarray of float
        The :math:`\beta` parameter, to define the behaviour of the backbone.
    rho : float or ndarray of float
        The Rho or correlation parameter, for correlations between asset price and volatility.
    nu : float or ndarray of float
        The Nu or stochastic volatility parameter.
    model : {'l', 'n', 'bn'}, optional
        Volatility model type: lognormal, normal, boundaed normal.
        l = lognormal (black) [default] | n = normal | bn = bounded normal
    boundary : float, optional
        Boundary for the bounded normal model. If model = 'bn', bound needs to be specified.

    Returns
    -------
    alpha : float or ndarray of float
        The :math:`\alpha` parameter for the SABR implementation.

    """

    # then move on to find alpha with the simplified formulas
    if model == 'l':
        print("l")
    elif model == 'n':
        print("n")
    else:
        raise NotImplementedError("Model not implemented in this function.")


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
        Volatility model type: lognormal, normal, boundaed normal.
        l = lognormal (black) [default] | n = normal

    Returns
    -------
    beta : float
        The regression-based estimate for the SABR :math:`\beta` parameter estimate.

    """

    # if fwd = strike the normal and lognormal formulas simplify substantially
    # and depend on time only on a small order. As a consequence, the relationship can be estimated via regression
    # on logarithms

    if model == 'l':
        print("l")
    elif model == 'n':
        print("n")
    else:
        raise NotImplementedError("Model not implemented in this function.")