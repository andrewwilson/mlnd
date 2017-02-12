from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import seaborn as sns

import os
import keras

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils
import pandas as pd
import numpy as np

def logreturn(px_latest, px_prev):
    return np.log(px_latest / px_prev)

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
    df = pd.DataFrame(index=px.index)
    for period in [1,2,4,8,16,32,64,128]:
        df['ret-{}'.format(period)] = np.tanh(0.33* (px - px.shift(period) )/ np.sqrt(period))
        df['ma{}-{}'.format(2*period, period)] = masig(px, 2*period, period)
        df['ewma{}-{}'.format(2*period, period)] = ewmasig(px, 2*period, period)
    df['vol'] = px.ewm(128).std()    
    return df


def target(price_series, units, categories, lookahead):
    y = logreturn(price_series, price_series.shift(-lookahead))
    y = y-y.mean()
    clip = (categories-1)/2
    y = clip + np.clip(y/units,-clip,clip).dropna().astype(int)
    return y

def preprocess(price_series, categories=3, units=0.0010, lookahead=10, vol_adjust=24*60*5):
    px = normalise_price(price_series, vol_adjust=vol_adjust)
    X = add_features(px).dropna()
    y = target(price_series, units=units, categories=categories, lookahead=lookahead).dropna()
    # align series after NA's removed
    idx = X.index.intersection(y.index)

    X = X.ix[idx]
    y = y.ix[idx]
    return X,y

def build_model(n_features, n_categories, loss='categorical_crossentropy'):
    model = Sequential()

    model.add(Dense(500, input_dim=n_features, activation='relu', init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(500, activation='relu', init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(500, activation='relu', init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(500, activation='relu', init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=n_categories, activation='softmax', init='he_normal'))

    model.compile(loss=loss, optimizer='adam')
