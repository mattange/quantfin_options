# -*- coding: utf-8 -*-
"""
SABR model related formulas.

The Stochastic Alpha Beta Rho model is the classic SABR model.
For references, see "Managing Smile Risk" by Hagan, Kumar, Lesniewski and Woodward (2002).

"""

import numpy as np


def sabr_vol(fwd, strike, tau, alpha, beta, rho, nu, model='l'):
    """
    Returns the SABR model implied volatility to be used in Black-Scholes formulas.

    Parameters
    ----------
    fwd : float or ndarray of float
        Forwards.
    strike : float or ndarray of float
        Strikes.
    tau : float or ndarray of float
        Times to expiry, normally in years assuming that volatilities are "per year equivalent".
    alpha : float or ndarray of float
        The Alpha parameter, mostly influencing overall option prices, regardless of skew behaviour.
    beta : float or ndarray of float
        The Beta parameter, to define the behaviour of the backbone.
    rho : float or ndarray of float
        The Rho or correlation parameter, for correlations between asset price and volatility.
    nu : float or ndarray of float
        The Nu or stochastic volatility parameter.
    model : {'l', 'n'}
        Volatility model type: lognormal, normal.
        l = lognormal (black) [default] | n = normal

    Returns
    -------
    sabr_vol : float or ndarray of float
        The volatility to use in the standard Black-Scholes formula.

    """


    logfoverk = np.log(fwd / strike)
    fk1_b = np.power(fwd * strike, 1 - beta)
    fk1_b2 = np.sqrt(fk1_b)
    zeta = nu / alpha * fk1_b2 * logfoverk
    xi = np.log((np.sqrt(1 - 2 * rho * zeta + zeta**2) + zeta - rho) / (1 - rho))
    sig = alpha / fk1_b2 / (1 + 1/24 * (1 - beta)**2 * logfoverk**2 + 1/1920 * (1 - beta)**4 * logfoverk**4) \
          * (zeta / xi) * (1 + (1/24 * (1 - beta)**2 * alpha**2 / fk1_b \
                                + 1/4 * rho * beta * nu * alpha / fk1_b2 \
                                + (2 - 3*rho**2)/24 * nu**2) * tau)
    if model == "l":
        return sig
    elif model == "n":
        raise NotImplementedError("Normal SABR model not yet implemented.")