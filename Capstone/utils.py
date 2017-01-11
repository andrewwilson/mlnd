
import pandas as pd
import numpy as np
import os
from collections import namedtuple
from functools import partial


def split_dataset(ds, *sizes):
    """ splits a dataframe into parts of relative size given by: sizes."""
    
    total_size = sum(sizes)
    total_items = len(ds)
    
    n_items = [ round(total_items * s / total_size) for s in sizes ]

    # compensate for rounding by adjusting the largest item
    largest_idx = n_items.index(max(n_items))
    n_items[largest_idx] += total_items - sum(n_items)
    
    assert sum(n_items) == total_items
    
    results = []
    last_idx = 0
    for n in n_items:
        idx = int(last_idx + n)
        results.append(ds[last_idx:idx])
        last_idx = idx
    
    return results

def load_stock_data(sym):
    df = pd.read_csv('stockData/{sym}.csv'.format(sym=sym), index_col=0, parse_dates=True)
    df['px'] = df['Adjusted Close']
    del df['Adjusted Close']
    return df


DataSet = namedtuple('DataSet', ['name', 'X_train', 'Y_train', 'X_dev', 'Y_dev', 'X_test', 'Y_test'])

def load_stock_datasets(features_and_targets_fn, train_frac=75, dev_frac=15, test_frac=15, sym_filter_fn=lambda x:True):
    res = {}
    files = os.listdir('stockData')
    for f in files:
        sym = f.split('.')[0]
        if sym_filter_fn(sym):
            raw = load_stock_data(sym)
            train, dev, test = split_dataset(raw, train_frac, dev_frac, test_frac)

            X_train, Y_train = features_and_targets_fn(train)
            X_dev, Y_dev = features_and_targets_fn(dev)
            X_test, Y_test = features_and_targets_fn(test)

            ds = DataSet(sym, X_train, Y_train, X_dev, Y_dev, X_test, Y_test)
            res[sym] = ds
    return res


def logreturn(px_latest, px_prev):
    return np.log(px_latest/px_prev)


def FT_logreturn_vs_logreturn(df, return_lookbacks=[1], target_lookaheads=[1]):
    """ features and targets function: 
        features: - log returns with various lookbacks.
        targets: - log return
    """
    results = pd.DataFrame(index=df.index)  
    feature_cols = []
    target_cols = []   
    for lb in return_lookbacks:
        col = 'lret-' + str(lb)
        results[col] = logreturn(df['px'], df['px'].shift(lb))
        feature_cols.append(col)     

    # add target feature to predict
    for la in target_lookaheads:
        col = 'target-' + str(la)
        results[col] = logreturn(df['px'].shift(-la), df['px'])
        target_cols.append(col)
        
    results = results.dropna() # so that features and targets are all complete, and have aligned samples   
    return results[feature_cols], results[target_cols]

def FT_ma_ewma_abs_logreturns_vs_abs_logreturn(df, ma_windows = [10], ewma_halflifes = [10]):
    """ features and targets function: 
        features: 
        - moving average of abs log return with various lookbacks.
        - ewma of abs log return with various lookbacks
        targets:
        - abs log return
    """
    results = pd.DataFrame(index=df.index)  
    feature_cols = []
    target_cols = []
    
    vol = logreturn(df['px'], df['px'].shift(1)).abs()
    results['vol'] = vol
    feature_cols.append('vol')
    future_vol = vol.shift(-1)
    for ma_win in ma_windows:
        col = 'ma-' + str(ma_win)
        ma = vol.rolling(ma_win).mean()
        results[col] = ma
        feature_cols.append(col)     
    
    for ewma_hl in ewma_halflifes:
        col = 'ewma-' + str(ewma_hl)
        ewma = vol.ewm(halflife=ewma_hl).mean()
        results[col] = ewma
        feature_cols.append(col)     

    # add target feature to predict
    results['target-1'] = future_vol
    target_cols.append('target-1')
        
    results = results.dropna() # so that features and targets are all complete, and have aligned samples   
    return results[feature_cols], results[target_cols]


def load_ds1():
    features_and_targets = partial(FT_logreturn_vs_logreturn, return_lookbacks=np.arange(40)+1, target_lookaheads=[1])
    return load_stock_datasets(features_and_targets)

def load_ds2():
    features_and_targets = partial(FT_ma_ewma_abs_logreturns_vs_abs_logreturn, ma_windows=np.arange(40)+1, ewma_halflifes=np.arange(40)+1)
    return load_stock_datasets(features_and_targets)