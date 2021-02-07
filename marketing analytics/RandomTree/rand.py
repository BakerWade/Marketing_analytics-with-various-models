import pandas as pd
import numpy as np

data = pd.read_csv('Marketing_updated.csv')

from sklearn.model_selection import train_test_split
X = data.drop('Response',axis=1)
y = data['Response']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

tree = DecisionTreeClassifier()
ran = RandomForestClassifier(n_estimators=200)

ran.fit(X_train,y_train)
pred = ran.predict(X_test)

from sklearn.metrics import r2_score, classification_report, confusion_matrix

print('R2:',r2_score(y_test,pred))
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))