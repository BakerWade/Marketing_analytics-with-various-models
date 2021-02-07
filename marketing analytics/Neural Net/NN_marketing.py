import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from tensorflow.keras import callbacks


data = pd.read_csv('Marketing_updated.csv')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X = data.drop('Response',axis=1).values
y = data['Response'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()

model.add(Dense(20, activation='relu'))
#model.add(Dropout())

#model.add(Dense(80, activation='relu'))
#model.add(Dropout(0.5))

model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=[tf.keras.metrics.BinaryAccuracy()])

early = EarlyStopping(monitor='val_loss', mode='min', verbose=2,patience=20)

model.fit(x=X_train,y=y_train,epochs=250, validation_data=(X_test,y_test), verbose=1, callbacks=[early])

#model.save('Market_model.keras')

model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
plt.show()

pred = (model.predict(X_test)>0.5).astype("int32")

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


