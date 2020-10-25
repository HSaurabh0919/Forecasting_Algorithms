

#Autoregression Problems

# Walk throught the test data, training and predicting 1 day ahead for all the test data
#Define df_training and df_test
resultsDict={}
predictionsDict={}
index = len(df_training)
yhat = list()
for t in tqdm(range(len(df_test.col))):
    temp_train = df[:len(df_training)+t]
    model = AR(temp_train.col)
    model_fit = model.fit()
    predictions = model_fit.predict(start=len(temp_train), end=len(temp_train), dynamic=False)
    yhat = yhat + [predictions]
    
yhat = pd.concat(yhat)
resultsDict['AR'] = evaluate(df_test.pollution_today, yhat.values)
predictionsDict['AR'] = yhat.values
