"""
Steps In Data Preprocessing:
1-Import the Libraries.
2-Import the dataset.
4-Dealing with Categorical Values.
3-Dealing with Missing Values.
5-Feature Scaling
6-Splitting the data.
"""
#1-import libraries
import pandas as pd
import numpy as np

#2- import datasets
veriler = pd.read_csv('iris.csv')
h=veriler.head(10)

#4-Dealing with Categorical Values.
#h.dropna(inplace = True) #dropna nan satırını kaldırır

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
spe=h.iloc[::,-1]
spe=le.fit_transform(spe)
spe=pd.DataFrame(spe,columns=['species'])
h=h.iloc[:,0:4]
h2=pd.DataFrame(h)
#3-Dealing with Missing Values.
from sklearn.preprocessing import Imputer
imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
h2=imp.fit_transform(h2)
h2=pd.DataFrame(h2)
sonuc=pd.concat([h2,spe],axis=1)
#one hot encıder kullanılabilir

#5-Feature Scaling
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()   
sonuc=ss.fit_transform(sonuc)#Veriye normalization uyguluyoruz

#6-Splitting the data.
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(sonuc,h,test_size=0.3,random_state=0)
























