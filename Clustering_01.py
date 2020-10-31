from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import datasets

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df=pd.read_csv(r"..\data.csv")

#Check the data
df.describe().T
df.isnull().sum()

#Using the KMeans Clsuetring 
model=KMeans(n_clusters=3).fit(df)
model.cluster_centers_

#Sample data for Predicting
a=[20,45]
b=[10,-241]
c=[10,17]
data=[a,b,c]
model.predict(data)


