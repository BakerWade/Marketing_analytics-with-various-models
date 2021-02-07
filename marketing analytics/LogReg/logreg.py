import numpy as np
import pandas as pd

data = pd.read_csv('Marketing_updated.csv')

X = data.drop('Response', axis=1)
y = data['Response']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=98)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)
pred = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, r2_score

print('R2_score:',r2_score(y_test,pred))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))