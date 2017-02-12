
from __future__ import division
import numpy as np
import pandas as pd

from utils import logreturn
# -------------------------
def ewmasig(x, lp, sp):
    volp = lp*5
    lma = x.ewm(span=lp).mean()
    sma = x.ewm(span=sp).mean()
    return np.tanh(15000 * (sma - lma) * lma.ewm(volp).std() / (np.sqrt(lp) *sma.ewm(volp).std()))

def masig(x, lp, sp):
    volp = lp*5
    lma = x.rolling(lp).mean()
    sma = x.rolling(sp).mean()
    return np.tanh(10000 * (sma - lma) * lma.rolling(volp).std() / (np.sqrt(lp) *sma.rolling(volp).std()))

def preprocess(price_series):
    px = logreturn(price_series, price_series.shift(1)).cumsum()
    ds = pd.DataFrame()
    ds['px'] = px
    for period in [2,4,8,16,32,64,128,256]:
        ds['lret-{}'.format(period)] = np.tanh(logreturn(px, px.shift(period)))
        ds['ma{}-{}'.format(2*period, period)] = masig(px, 2*period, period)
        ds['ewma{}-{}'.format(2*period, period)] = ewmasig(px, 2*period, period)
    return ds

# -------------------------

# how to use this model

class Model:
    def __init__(self, model):
        self.model = model

    def predict(self, price_series):
        ds = preprocess(price_series)

        prediction = self.model.predict(ds.as_matrix())
        return prediction


def load(filename):
    import keras
    model = keras.models.load_model(filename)
    return Model(model)



# filename = 'moving_average_500_500_500_500_20172901-194204.h5'
TRAINED_01 = 'moving_average_500_500_500_500_20172901-221323.h5'
