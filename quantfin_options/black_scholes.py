# -*- coding: utf-8 -*-
"""
Black-Scholes option pricing formulas.

The Lognormal model is the traditional Black-Scholes model and for reference see Wikipedia.
For the Normal and Bounded Normal model search for "Analytic Formula for the European Normal
Black Scholes Formula" by Kazuhiro Iwasawa (December 2001).

"""

import numpy as np
import math
from scipy.stats import norm
from scipy.optimize import newton

DTYPE = np.float


def option_fwd_value(fwd, strike, tau, sigma, opt_type='c', value_type='p', model='l', boundary=0.0):
    """
    Calculates the forward option price.
    In essence, this is equivalent to the standard option pricing if the discount rate is 0.
    No checks are done on the dimensions that need to be array_like or scalars in general and they need to be
    compatible for vector operations in numpy. No checks are also done on the values of the parameters
    (e.g. tau needs to be greater than 0, volatilities need to be positive ...).
    All values returned are forward values, i.e. need to be brought to present value if needed.

    Parameters
    ----------
    fwd : float or ndarray of float
        Forwards.
    strike : float or ndarray of float
        Strikes.
    tau : float or ndarray of float
        Times to expiry, normally in years assuming that volatilities are "per year equivalent".
    sigma : float or ndarray of float
        Volatilities :math:`\sigma`: 0.2 for 20% or in units of forward for normal models.
    opt_type : str or ndarray of str, optional
        Option type to price.
        c = call [default] | p = put | s = straddle
    value_type : str, optional
        Return value type: price, delta, gamma, vega, theta.
        Note that theta is the derivative with respect to time to expiry tau. I.e. for an increase in the time
        to expiry the value of the option increases. When considering the derivative with respect to time,
        change the sign of the output.
        p = price [default] | d = delta | g = gamma | v = vega | t = theta
    model : {'l', 'n', 'bn'}, optional
        Volatility model type: lognormal, normal or bounded normal.
        l = lognormal (black) [default] | n = normal | bn = bounded normal
    boundary : float, optional
        Boundary for the bounded normal model. If model = 'bn', bound cannot be None and needs to be specified.

    Returns
    -------
    fwd_price : float or array_like
        The option forward price based on the information provided.

    """

    sigsqrttau = sigma * np.sqrt(tau)

    if model == "l":
        d_plus = 1 / sigsqrttau * (np.log(fwd / strike) + 0.5 * sigsqrttau ** 2)
        d_minus = d_plus - sigsqrttau
        n_d_plus = norm.cdf(d_plus)
        n_d_minus = norm.cdf(d_minus)
        call_fwd_val = n_d_plus * fwd - n_d_minus * strike
        if (type(opt_type) is str) or (type(opt_type) is np.str_):
            if 'c' == opt_type:
                if value_type == "p":
                    return call_fwd_val
                elif value_type == "d":
                    return n_d_plus
                elif value_type == "g":
                    return norm.pdf(d_plus) / fwd / sigma / np.sqrt(tau)
                elif value_type == "v":
                    return fwd * norm.pdf(d_plus) * np.sqrt(tau)
                elif value_type == "t":
                    return fwd * norm.pdf(d_plus) * sigma / 2 / np.sqrt(tau)
            elif "p" == opt_type:
                if value_type == "p":
                    return call_fwd_val - fwd + strike
                elif value_type == "d":
                    return n_d_plus - 1
                elif value_type == "g":
                    return norm.pdf(d_plus) / fwd / sigma / np.sqrt(tau)
                elif value_type == "v":
                    return fwd * norm.pdf(d_plus) * np.sqrt(tau)
                elif value_type == "t":
                    return fwd * norm.pdf(d_plus) * sigma / 2 / np.sqrt(tau)
            elif 's' == opt_type:
                if value_type == "p":
                    return call_fwd_val + call_fwd_val - fwd + strike
                elif value_type == "d":
                    return 2 * n_d_plus - 1
                elif value_type == "g":
                    return 2 * norm.pdf(d_plus) / fwd / sigma / np.sqrt(tau)
                elif value_type == "v":
                    return 2 * fwd * norm.pdf(d_plus) * np.sqrt(tau)
                elif value_type == "t":
                    return 2 * fwd * norm.pdf(d_plus) * sigma / 2 / np.sqrt(tau)

        if type(opt_type) is np.ndarray:
            price = np.empty(shape=opt_type.shape, dtype=DTYPE)
            if value_type == "p":
                put_fwd_val = call_fwd_val - fwd + strike
            elif value_type == "d":
                call_fwd_val = n_d_plus
                put_fwd_val = n_d_plus - 1
            elif value_type == "g":
                call_fwd_val = norm.pdf(d_plus) / fwd / sigma / np.sqrt(tau)
                put_fwd_val = call_fwd_val
            elif value_type == "v":
                call_fwd_val = fwd * norm.pdf(d_plus) * np.sqrt(tau)
                put_fwd_val = call_fwd_val
            elif value_type == "t":
                call_fwd_val = fwd * norm.pdf(d_plus) * sigma / 2 / np.sqrt(tau)
                put_fwd_val = call_fwd_val
            price[opt_type == 'c'] = call_fwd_val[opt_type == 'c']
            price[opt_type == 'p'] = put_fwd_val[opt_type == 'p']
            price[opt_type == 's'] = call_fwd_val[opt_type == 's'] + put_fwd_val[opt_type == 's']
            return price

    elif model == "n":
        d1 = (fwd - strike) / sigsqrttau
        expd1 = np.exp(-d1 ** 2 / 2)
        if (type(opt_type) is str) or (type(opt_type) is np.str_):
            if 'c' == opt_type:
                nd1 = norm.cdf(d1)
                if value_type == "p":
                    return (fwd - strike) * nd1 + sigsqrttau * expd1 / math.sqrt(2 * math.pi)
                elif value_type == "d":
                    return nd1
                elif value_type == "g":
                    return expd1 / math.sqrt(2 * math.pi) / sigsqrttau
                elif value_type == "v":
                    return expd1 / math.sqrt(2 * math.pi) * np.sqrt(tau)
                elif value_type == "t":
                    return sigma * expd1 / 2 / math.sqrt(2 * math.pi * tau)
            elif "p" == opt_type:
                n_d1 = norm.cdf(-d1)
                if value_type == "p":
                    return (strike - fwd) * n_d1 + sigsqrttau * expd1 / math.sqrt(2 * math.pi)
                elif value_type == "d":
                    return -n_d1
                elif value_type == "g":
                    return expd1 / math.sqrt(2 * math.pi) / sigsqrttau
                elif value_type == "v":
                    return expd1 / math.sqrt(2 * math.pi) * np.sqrt(tau)
                elif value_type == "t":
                    return sigma * expd1 / 2 / math.sqrt(2 * math.pi * tau)
            elif 's' == opt_type:
                nd1 = norm.cdf(d1)
                n_d1 = norm.cdf(-d1)
                if value_type == "p":
                    return (fwd - strike) * nd1 + (strike - fwd) * n_d1 + 2 * sigsqrttau * expd1 / math.sqrt(2 * math.pi)
                elif value_type == 'd':
                    return nd1 - n_d1
                elif value_type == "g":
                    return 2 * expd1 / math.sqrt(2 * math.pi) / sigsqrttau
                elif value_type == "v":
                    return 2 * expd1 / math.sqrt(2 * math.pi) * np.sqrt(tau)
                elif value_type == "t":
                    return 2 * sigma * expd1 / 2 / math.sqrt(2 * math.pi * tau)
        if type(opt_type) is np.ndarray:
            price = np.empty(shape=opt_type.shape, dtype=DTYPE)
            nd1 = norm.cdf(d1)
            n_d1 = norm.cdf(-d1)
            if value_type == 'p':
                call_fwd_val = (fwd - strike) * nd1 + sigsqrttau * expd1 / math.sqrt(2 * math.pi)
                put_fwd_val = (strike - fwd) * n_d1 + sigsqrttau * expd1 / math.sqrt(2 * math.pi)
            elif value_type == "d":
                call_fwd_val = nd1
                put_fwd_val = -n_d1
            elif value_type == "g":
                call_fwd_val = expd1 / math.sqrt(2 * math.pi) / sigsqrttau
                put_fwd_val = expd1 / math.sqrt(2 * math.pi) / sigsqrttau
            elif value_type == "v":
                call_fwd_val = expd1 / math.sqrt(2 * math.pi) * np.sqrt(tau)
                put_fwd_val = expd1 / math.sqrt(2 * math.pi) * np.sqrt(tau)
            elif value_type == "t":
                call_fwd_val = sigma * expd1 / 2 / math.sqrt(2 * math.pi * tau)
                put_fwd_val = call_fwd_val
            price[opt_type == 'c'] = call_fwd_val[opt_type == 'c']
            price[opt_type == 'p'] = put_fwd_val[opt_type == 'p']
            price[opt_type == 's'] = call_fwd_val[opt_type == 's'] + put_fwd_val[opt_type == 's']
            return price

    elif model == "bn":
        raise NotImplementedError("Bounded normal model not yet implemented.")
    else:
        raise NotImplementedError("Model not implemented in this function.")


def option_impl_vol(fwd_price, fwd, strike, tau, opt_type, impl_vol_guess, model='l', boundary=0.0):
    """
    Calculates the implied option volatility based on the forward option price.
    No checks are done on the dimensions that need to be array_like or scalars and they need to be
    compatible for element-wise vector operations in numpy. No checks are also done on the values of the parameters
    (e.g. tau needs to be greater than 0, prices need to be positive ...).
    The root finding loop is executed over the size of the fwd_price input, therefore the sizes need to be consistent
    with the fwd_price input itself.

    Parameters
    ----------
    fwd_price : float or ndarray of float
        Forward option price from which the implied volatility will be derived.
    fwd : float or ndarray of float
        Forwards.
    strike : float or ndarray of float
        Strikes.
    tau : float or ndarray of float
        Times to expiry, normally in years assuming that volatilities are "per year equivalent".
    opt_type : str or ndarray of str
        Option type to price. Notably, to find implied volatilities the Put-Call parity will be used to convert
        in the money options to out of the money options where needed.
        c = call | s = straddle | p = put
    impl_vol_guess : float or ndarray of float
        Initial guess for the implied volatility.
    model : {'l', 'n', 'bn'}, optional
        Volatility model type: lognormal, normal or bounded normal.
        l = lognormal (black) [default] | n = normal | bn = bounded normal
    boundary : float, optional
        Boundary for the bounded normal model. If model = 'bn', bound cannot be None and needs to be specified.

    Returns
    -------
    implied_vol : float or ndarray of float
        The implied volatility based on the information provided.
    """

    # convert all inputs to arrays if they are scalars, so that the loop can be done in numpy
    # and copy them over where needed so that values later on do not get modified
    fwd_price_mod = np.atleast_1d(np.copy(np.asanyarray(fwd_price, dtype=DTYPE)))
    fwd_mod = np.atleast_1d(np.asanyarray(fwd, dtype=DTYPE))
    strike_mod = np.atleast_1d(np.asanyarray(strike, dtype=DTYPE))
    tau_mod = np.atleast_1d(np.asanyarray(tau, dtype=DTYPE))
    impl_vol_guess_mod = np.atleast_1d(np.asanyarray(impl_vol_guess, dtype=DTYPE))
    opt_type_mod = np.atleast_1d(np.copy(np.asanyarray(opt_type, dtype=np.str_)))

    # convert with put-call parity - assuming now that the dimensions are compatible
    # a) convert calls into puts where strike is lower than fwd
    idx = np.logical_and(opt_type == 'c', strike < fwd)
    fwd_price_mod[idx] = fwd_price_mod[idx] - fwd_mod[idx] + strike_mod[idx]
    opt_type_mod[idx] = 'p'
    # b) convert puts into calls where strike is higher than fwd
    idx = np.logical_and(opt_type == 'p', strike > fwd)
    fwd_price_mod[idx] = fwd_price_mod[idx] + fwd_mod[idx] - strike_mod[idx]
    opt_type_mod[idx] = 'c'

    impl_vol = np.empty(fwd_price_mod.shape, dtype=DTYPE)

    def func(sig, pos):
        return fwd_price_mod[pos] - option_fwd_value(fwd_mod[pos], strike_mod[pos],
                                                     tau_mod[pos], sig, opt_type_mod[pos], model=model,
                                                     boundary=boundary)

    for i in range(fwd_price_mod.shape[0]):
        impl_vol[i] = newton(func, impl_vol_guess_mod[i], args=(i,))

    if impl_vol.size == 1:
        impl_vol = np.asscalar(impl_vol)

    return impl_vol


def option_convert_impl_vol(fwd, strike, tau, sigma, model, new_model, boundary=0.0, new_boundary=0.0):
    """
    Converts the implied volatility between models.
    As models are different, the conversion is only valid for the specified conditions.

    Parameters
    ----------
    fwd : float or ndarray of float
        Forwards.
    strike : float or ndarray of float
        Strikes.
    tau : float or ndarray of float
        Times to expiry, normally in years assuming that volatilities are "per year equivalent".
    sigma : float or ndarray of float
        Volatilities :math:`\sigma`: 0.2 for 20% or in units of forward.
    model : {'l', 'n', 'bn'}
        Volatility model type: lognormal, normal or bounded normal.
        l = lognormal (black) | n = normal | bn = bounded normal
    boundary : float, optional
        Boundary for the bounded normal model. If model = 'bn', bound needs to be specified.
    new_model : {'l', 'n', 'bn'}, optional
        Volatility model type: lognormal, normal or bounded normal.
        l = lognormal (black) | n = normal | bn = bounded normal
    new_boundary : float, optional
        Boundary for the bounded normal model. If model = 'bn', bound needs to be specified.


    Returns
    -------
    implied_vol : float or ndarray of float
        The local implied volatility based on the information provided according to the new model.

    """

    if new_model == model:
        if model == 'bn' and boundary == new_boundary:
            return sigma
        if model != 'bn':
            return sigma

    if type(fwd) is float or type(fwd) is np.float:
        if fwd <= strike:
            opt_type = 'c'
        else:
            opt_type = 'p'
    if type(fwd) is np.ndarray:
        opt_type = np.array(['c' for i in range(fwd.size)], dtype=np.str_)
        opt_type[fwd > strike] = 'p'

    fwd_price = option_fwd_value(fwd, strike, tau, sigma, opt_type, value_type='p', model=model, boundary=boundary)
    if new_model == 'n' or new_model == 'bn':
        impl_vol_guess = np.sqrt(fwd*strike) * sigma
    elif new_model == 'l':
        impl_vol_guess = sigma / np.sqrt(fwd*strike)
    return option_impl_vol(fwd_price, fwd, strike, tau, opt_type, impl_vol_guess,
                           model=new_model, boundary=new_boundary)
