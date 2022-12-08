#!/usr/bin/env python
# coding: utf-8

# In[297]:


import pandas as pd 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import scipy
import scipy.stats
import datetime
from scipy.stats import uniform
import statistics
from statistics import multimode
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statistics import median
from math import isnan
from itertools import filterfalse


# In[194]:


DF=pd.read_excel("Desktop/TPML2/ONLINE_RETAILS.xlsx")


# In[196]:


DF


# In[198]:


print(DF)


# In[200]:


DF.dtypes


# In[202]:


DistribPays=DF.Country.value_counts()
DistribPays


# In[204]:


FreqPays=DF.Country.value_counts(normalize=True).round(6).mul(100)
FreqPays


# In[206]:


DistribProduit=DF.Description.value_counts()
DistribProduit


# In[208]:


FreqPays=DF.Description.value_counts(normalize=True).round(6).mul(100)
FreqPays


# #  Question 1

# In[212]:


DfVolume=DF["Quantity"].describe()
DfVolume


# In[214]:


DfCountry=DF.groupby('Country')
DfCountry["Quantity"].agg(['mean','median'])


# In[216]:


DfCount1=DfCountry["TotalPrice"].agg(['mean','median'])
DfCount1


# In[218]:


length = 10000
bins=500
df1=DfCountry["Quantity"].agg(['mean','median'])
y, x = np.histogram(df1, bins=bins, density=True)
x = (x + np.roll(x, -1))[:-1] / 2.0
plt.figure(figsize=(12,8))
plt.hist(df1, bins=500, density=True)
plt.title("Volume des produits acheté par les clients")
plt.show()


# # Question 2

# In[220]:


DfPrice=DF["TotalPrice"].describe()
DfPrice


# In[222]:


DfCountry["TotalPrice"].agg("mean","median")


# In[224]:


length = 10000
bins=500
df2=DfCountry["TotalPrice"].agg("mean","median")
y, x = np.histogram(df2, bins=bins, density=True)
x = (x + np.roll(x, -1))[:-1] / 2.0
plt.figure(figsize=(12,8))
plt.hist(df2, bins=500, density=True)
plt.title("Distribution des Montants par produits acheté par les clients")
plt.show()


# # Question 3
 1) filtrer 1er trimestre
 2) Top 5 produits vendus pendant le 1er trimestre
 3) filtrer 2eme semestre  
 4)Top 5 produits vendus pendant le 2eme trimestre
 5) 3eme trimestre
 6)Top 5 produits vendus pendant le 3eme trimestre
 7)4eme trimestre
 8)Top 5 produits vendus pendant le 4eme trimestre
# In[226]:


DFTrim_1=((DF['InvoiceDate']>='2011-01-01')&(DF["InvoiceDate"]<'2011-04-01'))
DFT1=DF.loc[DFTrim_1]
DFT1


# In[228]:


TopProd1=DFT1.Description.value_counts()[:5]
TopProd1


# In[230]:


DFTrim_2=((DF['InvoiceDate']>='2011-04-01')&(DF["InvoiceDate"]<'2011-07-01'))
DFT2=DF.loc[DFTrim_2]
DFT2


# In[232]:


TopProd2=DFT2.Description.value_counts()[:5]
TopProd2


# In[234]:


DFTrim_3=((DF['InvoiceDate']>='2011-07-01')&(DF["InvoiceDate"]<'2011-10-01'))
DFT3=DF.loc[DFTrim_3]
DFT3


# In[236]:


TopProd3=DFT3.Description.value_counts()[:5]
TopProd3


# In[238]:


DFTrim_4=((DF['InvoiceDate']>='2011-10-01')&(DF["InvoiceDate"]<'2012-01-01'))
DFT4=DF.loc[DFTrim_4]
DFT4


# In[240]:


TopProd4=DFT4.Description.value_counts()[:5]
TopProd4


# In[ ]:





# # Question 4 

# 1) filtrer par date et par pays 
# 
# 2) Top 5 des pays avec plus gros total d'achats 
# 

# In[242]:


Pays_T1=DFT1.groupby("Country")
Top_AchatPays= Pays_T1['TotalPrice'].sum()
Top_AchatPays.sort_values(ascending=False)[:5]


# In[244]:


DFT2011=((DF['InvoiceDate']>='2010-01-01')&(DF["InvoiceDate"]<'2011-12-01'))
DF2011=DF.loc[DFT2011]
DF2011


# In[246]:


Pays_2011=DF2011.groupby("Country")
Top_AchatPays_2= Pays_2011['TotalPrice'].sum()
Top_AchatPays_2.sort_values(ascending=False)[:5]


# In[ ]:





# # Question 5

# In[247]:


Top_Pays=Top_AchatPays.sort_values(ascending=False)[:5]
Top_Pays.describe()


# In[ ]:





# In[248]:


UK_filter = DF['Country'] == 'United Kingdom'
dfuk = DF.loc[UK_filter]
dfuk


# In[249]:


dfuk["CustomerID"].value_counts()
ClientUK = dfuk.groupby("CustomerID")
Topclientuk = ClientUK['TotalPrice'].agg(['sum'])
Topclientuk
TopCUK = Topclientuk.sort_values("sum", ascending=False).head(100)
TopCUK


# In[250]:


topUKclientfreq=TopCUK.describe()
topUKclientfreq


# In[251]:


countries= ['United Kingdom','Netherlands','EIRE','Germany','France']
DF3=DF[DF.Country.isin(countries)]
DF3


# In[ ]:


list(DF3)


# In[252]:


rolling_mean=DF3['Quantity'].rolling(window=12,center=False).mean()
rolling_mean


# In[253]:


plt.plot(rolling_mean)


# In[254]:


rolling_std=DF3['Quantity'].rolling(window=12,center=False).std()
rolling_std


# In[ ]:


plt.plot(rolling_std)

On peut remarquer que la moyenne mobile et l'écart-type augmentent avec le temps, on peut dire que cette série n'est pas stationnaire 

# In[261]:


result= adfuller(Date_Q["Quantity"])
print('Statistiques ADF : {}'.format(result[0]))
print('p-value : {}'.format(result[1]))
print('Valeurs Critiques :')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))
result    


# In[258]:


Date_Q=DF3[['InvoiceDate','Quantity']]
Date_Q

Modèle linéaire 1
# 2603=
# JUMBO BAG RED RETROSPOT                596
# WHITE HANGING HEART T-LIGHT HOLDER    520
# WHITE HANGING HEART T-LIGHT HOLDER    531
# JUMBO BAG RED RETROSPOT               421
# REGENCY CAKESTAND 3 TIER              535
# Produits qui se vendent le mieux 'best sellers'
# 

# In[294]:


# Générer des données aléatoires
x = np.random.rand(10000, 100)
y = 2603 * x + np.random.rand(10000, 100)
# Implémentation avec sckit-learn
# initialisation du modèle
regression_model = LinearRegression()
# Adapter les données (entraînement du modèle)
regression_model.fit(x, y)
# Prédiction
y_predicted = regression_model.predict(x)
# Évaluation du modèle
rmse = mean_squared_error(y, y_predicted)
r2 = r2_score(y, y_predicted)
# Affichage des valeurs
print("Pente : " ,regression_model.coef_)
print("Ordonnée à l'origine : ", regression_model.intercept_)
print("Racine carrée de l'erreur quadratique moyenne : ", rmse)
print('Sccore R2 : ', r2)
# Tracée des valeurs
# Points de données
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')
# Valeurs prédites
plt.plot(x, y_predicted, color='r')
plt.show()
y

Modele linéaire 2
1605496,6
United Kingdom    1402263.60
Netherlands         71959.76
EIRE                50113.89
Australia           40700.47
France              40458.88
Chiffre d'Affaire premiers semestre
# In[293]:


# Générer des données aléatoires
x = np.random.rand(10000, 100)
y = 1605496.6 * x + np.random.rand(10000,100)
# Implémentation avec sckit-learn
# initialisation du modèle
regression_model = LinearRegression()
# Adapter les données (entraînement du modèle)
regression_model.fit(x, y)
# Prédiction
y_predicted = regression_model.predict(x)
# Évaluation du modèle
rmse = mean_squared_error(y, y_predicted)
r2 = r2_score(y, y_predicted)
# Affichage des valeurs
print("Pente : " ,regression_model.coef_)
print("Ordonnée à l'origine : ", regression_model.intercept_)
print("Racine carrée de l'erreur quadratique moyenne : ", rmse)
print('Sccore R2 : ', r2)
# Tracée des valeurs
# Points de données
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')
# Valeurs prédites
plt.plot(x, y_predicted, color='r')
plt.show()
y


# In[ ]:





# In[ ]:




