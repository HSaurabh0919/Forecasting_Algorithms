1. # Autoregression Problems


# Walk throught the test data, training and predicting 1 day ahead for all the test data
#Define df_training and df_test
resultsDict={}
predictionsDict={}
index = len(df_training)
yhat = list()

for t in tqdm(range(len(df_test.col))):   # Tqdm makes you show your loop  a smart progress meter
    temp_train = df[:len(df_training)+t]
    model = AR(temp_train.col)
    model_fit = model.fit()
    predictions = model_fit.predict(start=len(temp_train), end=len(temp_train), dynamic=False)
    yhat = yhat + [predictions]
    
yhat = pd.concat(yhat)
resultsDict['AR'] = evaluate(df_test.pollution_today, yhat.values)
predictionsDict['AR'] = yhat.values




2.#Drawing the Correlation Matrix
def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure(figsize=(16,12))
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Wine data set features correlation\n',fontsize=15)
    labels=df.columns
    ax1.set_xticklabels(labels,fontsize=9)
    ax1.set_yticklabels(labels,fontsize=9)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[0.1*i for i in range(-11,11)])
    plt.show()

correlation_matrix(df)


3. # Variance finding using PCA(Principal Component Analysis)
from sklearn.decomposition import PCA
pca=PCA(n_components=None)
dfx_pca=pca.fit(dfx)

#Plot the Explained Variance Ratio

#https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Clustering-Dimensionality-Reduction/Principal%20Component%20Analysis.ipynb
plt.figure(figsize=(10,6))
plt.scatter(x=[i+1 for i in range(len(dfx_pca.explained_variance_ratio_))],y=dfx_pca.explained_variance_ratio_,s=200,alpha=0.75,c='orange',edgecolor='k')
plt.grid(True)
plt.title("Explained Varaiance Ratio of the \n fitted Principal Component Vector\n"fontsize=25)
plt.xlabel("Principal Components",fontsize=15)
plt.xticks[i+1 for i in range(len(dfx_pca.explained_variance_ratio_))],fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Explained variance ratio",fontsize=15)
plt.show()

#Transform the scaled data set using the fitted PCA object
dfx_trans = pca.transform(dfx)

dfx_trans = pd.DataFrame(data=dfx_trans)
dfx_trans.head(10)
#Plot the first two columns of this transformed data set with the color set to original ground truth class labe
plt.figure(figsize=(10,6))
plt.scatter(dfx_trans[0],dfx_trans[1],c=df['Class'],edgecolors='k',alpha=0.75,s=150)
plt.grid(True)
plt.title("Class separation using first two principal components\n",fontsize=20)
plt.xlabel("Principal component-1",fontsize=15)
plt.ylabel("Principal component-2",fontsize=15)
plt.show()

