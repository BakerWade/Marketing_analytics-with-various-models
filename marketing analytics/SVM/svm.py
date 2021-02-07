import numpy as np
import pandas as pd

data = pd.read_csv('Marketing_updated.csv')

X = data.drop('Response',axis=1)
y = data['Response']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=46)

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param = {'C':[1,10,100,1000,10000],'gamma':[1,0.1,0.01,0.001,0.0001],'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param, refit=True, verbose=2)



model = SVC()

grid.fit(X_train,y_train)

pred = grid.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, r2_score

print('R2_score:', r2_score(y_test,pred))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))