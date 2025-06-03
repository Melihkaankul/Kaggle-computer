import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        df=pd.read_csv("/kaggle/input/laptop-prices/laptop_prices.csv")

df=df[["Company","Product","TypeName","Inches","Ram","Price_euros","Screen"]]
df.columns=["Sirket","Urun","Tip","Inc","Ram","Euro_fiyati","Ekran"]

y = df["Euro_fiyati"]
x = df.drop("Euro_fiyati", axis=1)
kategorik_sutunlar = x.select_dtypes(include=['object']).columns
x = pd.get_dummies(x, columns=kategorik_sutunlar)
lr = LinearRegression()
model = lr.fit(x, y)
model.score(x,y)
ohe = OneHotEncoder()
kategorik_veriler = df[['Sirket', 'Urun', 'Tip',"Ekran"]]
ohe.fit(kategorik_veriler)
tahmin_array=[16,15.4]
aa=list(ohe.transform([["Apple","MacBook Pro","Ultrabook","Standard"]]).toarray()[0])
tahmin_array=tahmin_array+aa
model.predict([tahmin_array])
