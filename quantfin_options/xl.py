import xlwings as xw

from .black_scholes import fwd_value
from .black_scholes import impl_vol

impl_vol = xw.func(impl_vol)
fwd_value = xw.func(fwd_value)

def xl_rename(new_name):
    def decorator(fn):
        fn.__name__ = new_name
        return fn
    return decorator

#from .black_scholes import fwd_value as qfBSFwdValue
#from .black_scholes import impl_vol as qfBSImplVol

#qfBSImplVol = xw.func(qfBSImplVol)
#qfBSFwdValue = xw.func(qfBSFwdValue)

