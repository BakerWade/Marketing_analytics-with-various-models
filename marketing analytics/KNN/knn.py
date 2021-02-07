import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('Marketing_updated.csv')

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_data = scaler.fit_transform(data.drop('Response',axis=1))

feats_data = pd.DataFrame(scaled_data, columns=['Kidhome', 'Teenhome', 'MntWines', 'MntFruits', 'MntMeatProducts',
       'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
       'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
       'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3',
       'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2',
       'Complain', 'Dt_Customer_year', 'Income', 'Age', 'Divorced',
       'Married', 'Single', 'Together', 'Widow', '2n Cycle', 'Basic',
       'Graduation', 'Master', 'PhD', 'AUS', 'CA', 'GER', 'IND', 'ME', 'SA',
       'SP', 'US'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feats_data,data['Response'],test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier

Kn = KNeighborsClassifier(n_neighbors=14)

Kn.fit(X_train,y_train)
pred = Kn.predict(X_test)

from sklearn.metrics import r2_score, classification_report, confusion_matrix

"""error_rate = []

for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))"""

"""plt.plot(range(1,50),error_rate, color='red', ls='--', marker='o',markerfacecolor='blue',markersize=10)
plt.title('Error rate VS K_value')
plt.xlabel('K_value')
plt.ylabel('Error_rate')"""
#plt.show()
print('R2_score:',r2_score(y_test,pred))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
