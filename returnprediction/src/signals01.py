import pandas as pd
import numpy as np
from utils import logreturn

def masig(x, lp, sp):
    volp = lp*5
    lma = x.rolling(lp).mean()
    sma = x.rolling(sp).mean()
    return np.tanh((sma - lma) * lma.rolling(volp).std() / (np.sqrt(lp) *sma.rolling(volp).std()))

def ewmasig(x, lp, sp):
    volp = lp*5
    lma = x.ewm(span=lp).mean()
    sma = x.ewm(span=sp).mean()
    return  np.tanh((sma - lma) * lma.ewm(volp).std() / (np.sqrt(lp) *sma.ewm(volp).std()))

def normalise_price(p, vol_adjust=None):
    px = logreturn(p, p.shift(1)).cumsum()
    if vol_adjust:
        px = px / ewm_vol(px, com=vol_adjust)
    return px

def ewm_vol(p,com):
    return p.diff().ewm(com, min_periods=com).std()

def add_features(px):
    df = pd.DataFrame({'px':px})
    for period in [2,4,8,16,32,64,128,256]:
        df['ret-{}'.format(period)] = np.tanh(0.33* (px - px.shift(period) )/ np.sqrt(period))
        #df['ma{}-{}'.format(2*period, period)] = masig(px, 2*period, period)
        df['ewma{}-{}'.format(2*period, period)] = ewmasig(px, 2*period, period)
    return df

def preprocess(price_series):
    day = 24*60
    px = normalise_price(price_series, vol_adjust=5*day)
    df = add_features(px)
    return df

