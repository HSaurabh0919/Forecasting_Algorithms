import pandas as pd
import matplotlib.pyplot as plt


import seaborn as sns


import warnings
import numpy as np
import itertools
import matplotlib.pyplot as plt
import statsmodels.api as sm

import itertools


import scipy
import matplotlib.pyplot as plt

#from  sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import sklearn.model_selection as ms
import seaborn as sns
import matplotlib
from sklearn.preprocessing import Binarizer
import sklearn
import math
from sklearn.preprocessing import OneHotEncoder

#Defaults
plt.rcParams['figure.figsize']=(20.0,10.0)
plt.rcParams.update({'font.size':12})
plt.style.use('ggplot')

#Loading the data
df1=pd.read_csv(r"data\PVR.csv")
df2=pd.read_csv(r"data\IMAX.csv")

#Cleaning the data and removing unwanted features, we are only predicting the closing values
df1.drop(["Open","Volume","High","Low","Adj Close"],axis=1,inplace=True)
df2.drop(["Open","Volume","High","Low","Adj Close"],axis=1,inplace=True)
df3=pd.merge(df1, df2, on="Date")
target=df3["Close_x"]
df3 = df3.drop(["Date","Close_x"],axis = 1)


#Modelling the date
X_train,X_test,y_train,y_test=ms.train_test_split(df3,target,test_size=0.4,random_state=42)

#Standard Scaling to bring the data to mean zero 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#Polynomial Regression
poly_reg = PolynomialFeatures(degree = 4,interaction_only=False, include_bias=True)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_train, y_train)
lin_reg_1 = linear_model.LassoLars(alpha=0.009,max_iter=200)
lin_reg_1.fit(X_poly, y_train)
# predicitng 
pred_val = lin_reg_1.predict(poly_reg.fit_transform(X_test))
print(r2_score(y_test, pred_val))
mse = sklearn.metrics.mean_squared_error(y_test, pred_val)
rmse = math.sqrt(mse)
print(rmse)

#ElasticSearch
from sklearn.linear_model import ElasticNet

model = ElasticNet()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(r2_score(y_test, pred))
mse = sklearn.metrics.mean_squared_error(y_test, pred)
rmse = math.sqrt(mse)
print(rmse)

#Lasso Regression
from sklearn.linear_model import Lasso
model = Lasso()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(r2_score(y_test, pred))
mse = sklearn.metrics.mean_squared_error(y_test, pred)
rmse = math.sqrt(mse)
print(rmse)

#Ridge Regression
from sklearn.linear_model import Ridge
model = Ridge()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(r2_score(y_test, pred))
mse = sklearn.metrics.mean_squared_error(y_test, pred)
rmse = math.sqrt(mse)
print(rmse)


#Ransac  Regression
from sklearn.linear_model import RANSACRegressor
model = RANSACRegressor()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(r2_score(y_test, pred))
mse = sklearn.metrics.mean_squared_error(y_test, pred)
rmse = math.sqrt(mse)
print(rmse)


#XGBoost
from xgboost import XGBRegressor
my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(X_train, y_train, verbose=False)
# make predictions
pred = my_model.predict(X_test)
print(r2_score(y_test, pred))
mse = sklearn.metrics.mean_squared_error(y_test, pred)
rmse = math.sqrt(mse)
print(rmse)


