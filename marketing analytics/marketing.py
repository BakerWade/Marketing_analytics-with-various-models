import numpy as np
from numpy.core.arrayprint import IntegerFormat 
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.series import Series
import seaborn as sns
from seaborn import palettes

df = pd.read_csv('marketing_data_original.csv')
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
df['Dt_Customer_month'] = df['Dt_Customer'].apply(lambda time: time.month)
df['Dt_Customer_year'] = df['Dt_Customer'].apply(lambda time: time.year)
df = df.drop(['ID','Dt_Customer','Recency'],axis=1)

df['Income']=df[' Income '].replace( '[\$,)]','', regex=True ).astype(float)
df = df.drop(' Income ',axis=1)

df['Age'] = 2020 - df['Year_Birth']
df = df.drop('Year_Birth',axis=1)

df['Marital_Status'] = df['Marital_Status'].replace(['YOLO','Absurd','Alone'],'Single')
df['Income'] = df['Income'].replace(666666.0,62653.0)

df_income = df.corr()['Income'].sort_values(ascending=False)
null = df.loc[df.isnull().any(axis=1)]
avg_income = df.groupby('NumCatalogPurchases').mean()['Income']

"""avg_df = pd.merge(null,avg_income,on='Age')
avg_df = avg_df[['Age','Income_y']]
avg_income2 = {77:62324.166667, 69:55999.857143, 66:58628.387755, 65:57611.375000, 63:54020.268293, 62:58603.711538, 61:56324.080000, 59:57161.057143, 57:48815.863636, 56:56473.902439, 51:51205.628571, 52:52395.973333, 48:51123.333333, 47:47219.722222, 42:46012.434211, 39:46894.921053, 38:52683.795455, 37:47996.390244, 34:43346.414634, 31:42250.172414}"""

def fill_income(NumCatalogPurchases,Income):
    if np.isnan(Income):
        return avg_income[NumCatalogPurchases]
    else:
        return Income

df['Income'] = df.apply(lambda x: fill_income(x['NumCatalogPurchases'], x['Income']), axis=1)

dum = pd.get_dummies(df['Marital_Status'])
dum2 = pd.get_dummies(df['Education'])
dum3 =  pd.get_dummies(df['Country'])

df = pd.concat([df.drop(['Marital_Status','Education','Country','Dt_Customer_month'],axis=1),dum,dum2,dum3],axis=1)

#print(df.info())
#sns.jointplot(x='MntGoldProds',y='Income', data=df, hue='AcceptedCmp5',palette='viridis',kind='scatter')

#print(df.groupby('Age').mean()['Income'])


"""sns.heatmap(df4.corr(), annot=True, cmap= 'viridis')
plt.show()"""

#df.to_csv('Marketing_updated.csv',index=False)

#df2 = pd.read_csv('Marketing data2.csv')




