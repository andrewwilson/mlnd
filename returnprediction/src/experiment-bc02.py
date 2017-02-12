import pandas as pd

from model02 import preprocess, build_model
from training import train_model

all_data = pd.read_hdf('trades.bitstamp.resampled.30.h5')
train = all_data[:'2015']
dev = all_data['2016-01':'2016-06']
test = all_data['2016-07':]

print "train", len(train)
print "dev", len(dev)
print "test", len(test)


RUN_ID = "BC_02_"
SAMPLE_SECS=30
DAY = 24*60*60/SAMPLE_SECS
LOOKAHEAD=10
CATEGORIES=3
UNITS = 0.0010
VOL_ADJUST=5*DAY

EPOCHS=20
LIM=1000

X_train, y_train = preprocess(train['price'], categories=CATEGORIES,units=UNITS, lookahead=LOOKAHEAD, vol_adjust=VOL_ADJUST )
X_dev, y_dev = preprocess(dev['price'], categories=CATEGORIES,units=UNITS, lookahead=LOOKAHEAD, vol_adjust=VOL_ADJUST)

if LIM:
    X_train = X_train[:LIM]
    y_train = y_train[:LIM]
    X_dev = X_dev[:LIM]
    y_dev = y_dev[:LIM]

model = build_model(X_train.shape[1], n_categories=CATEGORIES, loss='categorical_crossentropy')


train_model(RUN_ID, model, epochs=EPOCHS)

from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


def report_performance():
    y_pred = model.predict(X_dev.as_matrix(), batch_size=2000, verbose=2)
    pred_classes = np_utils.categorical_probas_to_classes(y_pred)
    conf_matrix = confusion_matrix(y_dev, pred_classes)
    print conf_matrix
    print "accuracy", accuracy_score(y_dev, pred_classes)
    print "f1", f1_score(y_dev, pred_classes, average='weighted')
    #sns.heatmap(conf_matrix)
    #plt.show()

