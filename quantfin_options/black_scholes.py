# -*- coding: utf-8 -*-
"""
Black-Scholes option pricing formulas.

"""

from scipy.stats import norm
import numpy as np
import math


def option_fwd_value(fwd, strike, tau, sig, opt_type='c', model='l', bound = None):
    """
    Calculates the forward option price.
    No checks are done on the dimensions that need to be array_like
    or scalars in general or on the values (e.g. tau needs to be greater than 0, volatilities need to be positive ...).

    Parameters
    ----------
    :param fwd: float (scalar or array_like)
        Forwards.
    :param strike: float or array_like
        Strikes.
    :param tau: float (scalar or array_like)
        Times to expiry, normally in years assuming that volatilities are "per year equivalent".
    :param sig: float (scalar or array_like)
        Volatilities: 0.2 for 20% or in units of forward.
    :param opt_type: string (scalar or array_like), optional
        Option type to price.
        c = call [default]  |  p = put  |  s = straddle
    :param model: string, optional
        Volatility model type: lognormal, normal or bounded normal.
        l = lognormal (black) [default]  |  n = normal  |  bn = bounded normal
    :param bound: float (scalar or array_like) or None
        Boundary for the bounded normal model. If model = 'bn', bound cannot be None and needs to be specified.

    Returns
    -------
    :return fwd_price : float or array_like
        The option forward price based on the information provided.

    """

    sigsqrttau = sig * np.sqrt(tau)

    if model == "l":
        d_plus = 1 / sigsqrttau * (np.log(fwd / strike) + 0.5 * sigsqrttau ** 2)
        d_minus = d_plus - sigsqrttau
        n_d_plus = norm.cdf(d_plus)
        n_d_minus = norm.cdf(d_minus)
        call_fwd_val = n_d_plus * fwd - n_d_minus * strike
        if type(opt_type) is str:
            if 'c' == opt_type:
                return call_fwd_val
            elif "p" == opt_type:
                return call_fwd_val - fwd + strike
            elif 's' == opt_type:
                return call_fwd_val + call_fwd_val - fwd + strike
        if type(opt_type) is np.ndarray:
            price = np.empty(shape=opt_type.shape)
            price[opt_type == 'c'] = call_fwd_val[opt_type == 'c']
            price[opt_type == 'p'] = call_fwd_val[opt_type == 'p'] - fwd[opt_type == 'p'] + strike[opt_type == 'p']
            price[opt_type == 's'] = call_fwd_val[opt_type == 's'] + call_fwd_val[opt_type == 's'] \
                                     - fwd[opt_type == 's'] + strike[opt_type == 's']
            return price
    elif model == "n":
        d1 = (fwd - strike) / sigsqrttau
        expd1 = np.exp(-d1**2 / 2)
        if type(opt_type) is str:
            if 'c' == opt_type:
                nd1 = norm.cdf(d1)
                return (fwd - strike) * nd1 + sigsqrttau * expd1 / math.sqrt(2 * math.pi)
            elif "p" == opt_type:
                n_d1 = norm.cdf(-d1)
                return (strike - fwd) * n_d1 + sigsqrttau * expd1 / math.sqrt(2 * math.pi)
            elif 's' == opt_type:
                nd1 = norm.cdf(d1)
                n_d1 = norm.cdf(-d1)
                return (fwd - strike) * nd1 + (strike - fwd) * n_d1 + 2 * sigsqrttau * expd1 / math.sqrt(2 * math.pi)
        if type(opt_type) is np.ndarray:
            price = np.empty(shape=opt_type.shape)
            nd1 = norm.cdf(d1)
            call_fwd_val = (fwd - strike) * nd1 + sigsqrttau * expd1 / math.sqrt(2 * math.pi)
            n_d1 = norm.cdf(-d1)
            put_fwd_val = (strike - fwd) * n_d1 + sigsqrttau * expd1 / math.sqrt(2 * math.pi)
            price[opt_type == 'c'] = call_fwd_val[opt_type == 'c']
            price[opt_type == 'p'] = put_fwd_val[opt_type == 'p']
            price[opt_type == 's'] = call_fwd_val[opt_type == 's'] + put_fwd_val[opt_type == 's']
            return price
    elif model == "bn":
        raise NotImplementedError("Bounded normal model not yet implemented!")


def option_impl_vol(fwd, strike, tau, fwd_price, opt_type='c', model='l'):
    """
    Calculates the implied option volatility based on the forward option price.
    No checks are done on the dimensions that need to be array_like
    or scalars in general or on the values (e.g. tau needs to be greater than 0, prices need to be positive ...).

    :param fwd: float (scalar or array_like)
        Forwards.
    :param strike: float or array_like
        Strikes.
    :param tau: float (scalar or array_like)
        Times to expiry, normally in years assuming that volatilities are "per year equivalent".
    :param fwd_price: float (scalar or array_like)
        Forward option price from which the implied volatility will be derived.
    :param opt_type:

    :param model:
    :return:
    """

    if model == 'l':
    elif model == 'n':
    elif model == 'bn':

