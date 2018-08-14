# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Loading Libraries; handing most data as Pandas Dataframes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataDefs as dd

#from scikit-learn, using
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

#Seaborn for correlation heatmap visualization
import seaborn as sns

#Reading in the training .csv file; specify file location here along with test 
#data and test benchmark labels
df = pd.read_csv('/Users/avihiriyanna/Desktop/all/train.csv')
testDF = pd.read_csv('/Users/avihiriyanna/Desktop/all/test.csv')
benchmark = pd.read_csv('/Users/avihiriyanna/Desktop/all/sample_submission.csv')

#Extract column names for use in loop iterations here
columnNames = df.columns

#Trying to discretize string data into numbers for both the training and test set
numDescriptors = []
qualDescriptors = []

numericTable = []
numericTableTest =[]

#discretize non-numeric features into numbers by enumerating the unique values
#then create a new training and test set for model development. handling purely
#numeric data will make this process easier
for i in range(len(columnNames)):
    
    currData = df[columnNames[i]]
    
    if columnNames[i]!='SalePrice':
        currTest=testDF[columnNames[i]]
    
    #Compile Descriptors for posterity in a separate variable
    curDesc = df[columnNames[i]].describe()
    
    if len(curDesc) == 4:
        qualDescriptors.append(curDesc)
        
        labeledList, uniques = pd.factorize(currData)
        labeledList = pd.Series(labeledList, name=columnNames[i])
        numericTable.append(labeledList.to_frame())
        
        if columnNames[i]!='SalePrice':
            testLabel, testUniques = pd.factorize(currTest)
            testLabel = pd.Series(testLabel,name=columnNames[i])
            numericTableTest.append(testLabel.to_frame())

    else:
        numDescriptors.append(curDesc)
        
        currData = currData.replace(np.nan,0,regex=True) 
        numericTable.append(currData)
        
        if columnNames[i]!='SalePrice':
            currTest = currTest.replace(np.nan,0,regex=True)
            numericTableTest.append(currTest)
        
#Concatenate descriptors into a dataframe for easy vieweing and analysis        
qualSummary = pd.concat(qualDescriptors,axis=1)
numSummary = pd.concat(numDescriptors,axis=1)

fullTable = pd.concat(numericTable,axis=1)
fullTestTable = pd.concat(numericTableTest,axis=1)

#New Tables with numeric values only
newTable = fullTable.drop(['Id'],axis=1)
newTableFeats = newTable.drop(['SalePrice'],axis=1)
newTableTarget = newTable['SalePrice']

newTestTable=fullTestTable.drop(['Id'],axis=1)#remove the ID feature, its useless

#Normalize the data using StandardScaler
scaler = StandardScaler()

tableScaler = scaler.fit(newTableFeats)
testScaler = scaler.fit(newTestTable)

newTableNorm = tableScaler.transform(newTableFeats)
newTestNorm = testScaler.transform(newTestTable)

#Using quartiles for sales price using the describe method; use values to label
#houses by quartiles
salePriceArray = newTableTarget
benchmarkArray = benchmark['SalePrice']

nuLabels = dd.quartileLabeler(salePriceArray)
nuBenchmark = dd.quartileLabeler(benchmarkArray)


#Run a brief correlation analysis for feature correlations
corr = fullTable.corr()

#f,ax=plt.subplots(figsize=(12,9))
sns.heatmap(corr,vmax=.8,square=True)

#Determining descritization for labeling of house prices using clustering

#Kmeans
kmeans = KMeans(n_clusters =4)
kmeans.fit(newTableNorm)
y_km = kmeans.fit_predict(newTableNorm)

testKmeans = KMeans(n_clusters=4)
testKmeans.fit(newTestNorm)
testKLabs = testKmeans.fit_predict(newTestNorm)


#Labels for Classifications (quartile Labels,clustering labels, original saleprice labels)
compiledLabels = [nuLabels,y_km,newTableTarget]
testLabels = [nuBenchmark,testKLabs,benchmarkArray]

np.concatenate(compiledLabels)

#Feature Selection

#feature importance
featModel = ExtraTreesClassifier()
featModel.fit(newTableNorm,newTableTarget)
featureImportances = [featModel.feature_importances_,newTableFeats.columns]


#Test ANN (Original Labels)
#X = newTableNorm
#Y = newTableTarget
#Z = benchmarkArray
#
#modelOne = MLPClassifier(solver='sgd', max_iter=1500,verbose=10,hidden_layer_sizes=(79,79),learning_rate='adaptive')
#modelOne.fit(X,Y)
#predictOne = modelOne.predict(newTestNorm)
#
#plt.plot(predictOne-Z)

#Test ANN (KMeans Labels)
#X = newTableNorm
#Y = y_km
#Z = testKLabs
#
#modelOne = MLPClassifier(solver='sgd', verbose=10,max_iter=1500, hidden_layer_sizes=(79,79),learning_rate='adaptive')
#modelOne.fit(X,Y)
#predictOne = modelOne.predict(newTestNorm)
##
##plt.plot(predictOne-Z)
#
##Test ANN (quartile Labels)
#X = newTableNorm
#Y = nuLabels
#Z = nuBenchmark
#
#modelOne = MLPClassifier(solver='sgd', verbose=10,max_iter=1500, hidden_layer_sizes=(79,79),learning_rate='adaptive')
#modelOne.fit(X,Y)
#predictOne = modelOne.predict(newTestNorm)
#
#plt.plot(predictOne-Z)