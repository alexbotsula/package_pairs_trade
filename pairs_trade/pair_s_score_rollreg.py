
from collections import deque
import pandas as pd
import numpy as np
import statsmodels.api as sm



class PairBetaRollreg:
    def __init__(self):
        self._mod = None
        return

    def fit(self, r1, r2):
        X = sm.add_constant(r2)
        self._mod = sm.OLS(r1, X).fit()
        return self._mod

    @property 
    def residuals(self):
        return self._mod.resid



class PairOUResiduals:
    def __init__(self):
        self._m = None
        self._var = None
        self._sigma_eq = None
        self._cumm_resid = None
        return


    def fit(self, reg_resid):
        # Define OU process parameters
        self._cumm_resid = reg_resid.cumsum()
    
        resid_lag = self._cumm_resid.shift(1)
        resid_mod = sm.OLS(self._cumm_resid[1:], sm.add_constant(resid_lag[1:])).fit()
    
        # Calibrate OU - moments matching
        a, b = resid_mod.params
        self._m = a / (1 - b)
        var = np.nanvar(resid_mod.resid)
        self._sigma_eq = (var / (1 - b**2)) ** .5 


    @property
    def m(self):
        return self._m


    @property 
    def sigma_eq(self):
        return self._sigma_eq


    @property
    def cumm_resid(self):
        return self._cumm_resid



class PairSScoreRollreg:
    def __init__(self, window_size):
        self._asset1 = deque(maxlen=window_size+1)
        self._asset2 = deque(maxlen=window_size+1)
        self._window_size = window_size

        self._beta = PairBetaRollreg()
        self._ou = PairOUResiduals()

        self._sscore = None


    def step(self, asset1_close, asset2_close):
        self._asset1.append(asset1_close)
        self._asset2.append(asset2_close)

        if len(self._asset1) < self._window_size:
            return self

        r1 = np.log(pd.Series(self._asset1).pct_change()+1)[1:]
        r2 = np.log(pd.Series(self._asset2).pct_change()+1)[1:]

        self._beta.fit(r1, r2)
        self._ou.fit(self._beta.residuals)

        self._sscore = (self._ou.cumm_resid - self._ou.m) / self._ou.sigma_eq

        return self

    
    @property
    def sscore(self):
        if len(self._asset1) < self._window_size + 1:
            return 0
           
        return self._sscore.iloc[-1]

    @property 
    def beta(self):
        if len(self._asset1) < self._window_size + 1:
            return None
        
        return list(self._beta._mod.params)[1]
