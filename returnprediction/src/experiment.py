
# experiment:
# datasets: src
# model_definition:
# - preprocessor (features, augment-with-inverse)
# - objective
# function: lookahead.categories.categorisation.de - mean
# - architecture, loss
# function, optimiser
#
# model = train: (id, dataset, model_definition)
#
# test: evaluate: src + model, on
# metrics

from __future__ import division


import subprocess
import utils
#def categorical(n_features, nodes, dropout, loss, initialisation)

import signals01

print subprocess.check_output("pwd")

class Fx1MinBarLoader():
    def __init__(self, years, syms, fields):
        self.years = years
        self.syms = syms
        self.fields = fields

    def __call__(self):
        dfs = []
        for sym in self.syms:
            for year in self.years:
                df = utils.load_1minute_fx_bars(sym, year)[self.fields]
                dfs.append(df)
        return dfs


#preprocessor = signals01.preprocess
#objective = signals01.fut_log_return('px')

# sources = Fx1MinBarLoader(['EURUSD'], [2012,2013], ['close'])()
#
# print len(sources)
# for src in sources:
#     print src.tail()
#
#
# X = preprocessor(src)
# y = objective(src)
# print X.tail()

class Experiment:
    def __init__(self):
        pass

    def run(self):
        sources = self.load_data()
        for src in sources:
            X = self.preprocessor(src)
            y = self.objective(src)




#exp1 = Experiment(fx_1min_bar_loader(["EURUSD"], [2009,2010]))

import os
import env
import numpy as np
#----------------------------
import model01

file = os.path.join(env.SAVES_DIR, model01.TRAINED_01)
print file, os.path.exists(file)

model = model01.load(os.path.join(env.SAVES_DIR, model01.TRAINED_01))
prices = utils.load_1minute_fx_bars('XAUUSD', 2015)['close']
predictions = model.predict(prices)

returns = utils.logreturn(prices, prices.shift(1))
fut_return = returns.shift(-10)


print "prices", prices.shape
print "returns", returns.shape
print "fut_return", fut_return.shape
print "predictions", predictions.shape

pred_categories = np.argmax(predictions, axis=1)
for i in range(5):
    print i, 1e4*(fut_return * (pred_categories == i)).mean()


