#!usr/bin/env/python 
# -*- coding: utf-8 -*

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

def _load_data(data):
"""
data should be pd.DataFrame()
"""
n_prev = 10
docX, docY = [], []
for i in range(len(data)-n_prev):
    docX.append(data.iloc[i:i+n_prev].as_matrix())
    docY.append(data.iloc[i+n_prev].as_matrix())
if not docX:
    pass
else:
    alsX = np.array(docX)
    alsY = np.array(docY)
    return alsX, alsY

X, y = _load_data(df_test)

X_train = X[:25]
X_test = X[25:]

y_train = y[:25]
y_test = y[25:]

in_out_neurons = 2
hidden_neurons = 300
model = Sequential()
model.add(LSTM(in_out_neurons, hidden_neurons, return_sequences=False))
model.add(Dense(hidden_neurons, in_out_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")
model.fit(X_train, y_train, nb_epoch=10, validation_split=0.05)

predicted = model.predict(X_test)