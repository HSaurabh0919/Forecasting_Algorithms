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



#The pickle module implements binary protocols for serializing and de-serializing a Python object structure.
with open('clustering_model.pkl','wb') as f:
    pickle.dump(model,f)
    
 with open('clustering_model.pkl','rb') as f:
     resurrected_model=pickle.load(f)
    
resurrected_model.predict(data)




#Cluster_Centres with fixed random state
model_setseed=KMeans(n_clusters=4,random_state=10).fit(df)

model_setseed.cluster_centers_


#Practice code goes here
model5 = KMeans(n_clusters=15, random_state=10).fit(df)
sorted(model5.cluster_centers_.tolist())






#Elbow Method to find optimized  number of clusters
# Specifying the dataset and initializing variables
X = df
distorsions = []

# Calculate SSE for different K
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=301)
    kmeans.fit(X)
    distorsions.append(kmeans.inertia_)

# Plot values of SSE
plt.figure(figsize=(15,8))
plt.subplot(121, title='Elbow curve')
plt.xlabel('k')
plt.plot(range(2, 10), distorsions)
plt.grid(True)





#Silhouette Coefficient for better selection
X=df
silhouette_plot=[]
for k in range(2,10):
    clusters=KMeans(n_clusters=k,random_state=10)
    cluster_labels=clusters.fit_predict(X)
    silhouette_avg=metrics.silhouette_score(X,cluster_labels)
    silhouette_plot.append(silhouette_avg)

    
#Plot the Silhouette coefficientabs
plt.figure(figsize=(15,8))
plt.subplot(121,title='Silhouette Coefficients over k')
plt.xlabel('k')
plt.ylabel('Silhouette Coefficient')
plt.plot(range(2, 10), silhouette_plot)
plt.axhline(y=np.mean(silhouette_plot),color="red",linestyle="--")
plt.grid(True)



