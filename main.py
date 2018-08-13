# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Loading Libraries; handing most data as Pandas Dataframes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.neural_network import MLPClassifier

import seaborn as sns

#Reading in the training .csv file; specify file location here
df = pd.read_csv('/Users/avihiriyanna/Desktop/all/train.csv')
testDF = pd.read_csv('/Users/avihiriyanna/Desktop/all/test.csv')
benchmark = pd.read_csv('/Users/avihiriyanna/Desktop/all/sample_submission.csv')

#Extract column names for use in loop iterations here
columnNames = df.columns

#Trying to discretize string data into numbers
numDescriptors = []
qualDescriptors = []

numericTable = []
numericTableTest =[]

ncols = 9
nrows = int(np.ceil(len(columnNames)/(1.0*ncols)))

for i in range(len(columnNames)):
    
    currData = df[columnNames[i]]
    
    if columnNames[i]!='SalePrice':
        currTest=testDF[columnNames[i]]
    
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
        
        
qualSummary = pd.concat(qualDescriptors,axis=1)
numSummary = pd.concat(numDescriptors,axis=1)

fullTable = pd.concat(numericTable,axis=1)
fullTestTable = pd.concat(numericTableTest,axis=1)

#New Tables with numeric values only
newTable = fullTable.drop(['Id'],axis=1)
newTableFeats = newTable.drop(['SalePrice'],axis=1)
newTableTarget = newTable['SalePrice']

newTestTable=fullTestTable.drop(['Id'],axis=1)


newTableNorm = (newTableFeats-newTableFeats.mean())/(newTableFeats.max()-newTableFeats.min())
newTestNorm = (newTestTable-newTestTable.mean())/(newTestTable.max()-newTestTable.min())

#quartileLabels
salePriceArray = newTableTarget
benchmarkArray = benchmark['SalePrice']

nuLabels =[]
nuBenchmark = []

#NEED TO MAKE METHOD FOR THIS
for i in range(len(salePriceArray)):
    
    if salePriceArray.iloc[i]<=129975:
        nuLabels.append(0)
    elif salePriceArray.iloc[i]>129975 and salePriceArray.iloc[i]<=163000:
        nuLabels.append(1)
    elif salePriceArray.iloc[i]>163000and salePriceArray.iloc[i]<=214000:
        nuLabels.append(2)
    elif salePriceArray.iloc[i]>214000:
        nuLabels.append(3)

for i in range(len(benchmarkArray)):
    
    if benchmarkArray.iloc[i]<=129975:
        nuBenchmark.append(0)
    elif benchmarkArray.iloc[i]>129975 and benchmarkArray.iloc[i]<=163000:
        nuBenchmark.append(1)
    elif benchmarkArray.iloc[i]>163000and benchmarkArray.iloc[i]<=214000:
        nuBenchmark.append(2)
    elif benchmarkArray.iloc[i]>214000:
        nuBenchmark.append(3)

#correlation analysis
corr = fullTable.corr()

f,ax=plt.subplots(figsize=(12,9))
sns.heatmap(corr,vmax=.8,square=True)

#Determining descritization for labeling of house prices using various clustering methods

#Kmeans
kmeans = KMeans(n_clusters =4)
kmeans.fit(newTableNorm)
y_km = kmeans.fit_predict(newTableNorm)

testKmeans = KMeans(n_clusters=4)
testKmeans.fit(newTestNorm)
testKLabs = testKmeans.fit_predict(newTestNorm)

#
#fig, ax = plt.subplots(10,8,figsize=(50,50))
#
#xx = 0

#for i in range(10):
#    for j in range(8):
#        ax[i,j].scatter(newTableNorm.iloc[y_km == 0,79], newTableNorm.iloc[y_km ==0,xx], s=5, c='red')
#        ax[i,j].scatter(newTableNorm.iloc[y_km == 1,79], newTableNorm.iloc[y_km ==1,xx],  s=5, c='black')
#        ax[i,j].scatter(newTableNorm.iloc[y_km == 2,79], newTableNorm.iloc[y_km ==2,xx],  s=5, c='blue')
#        ax[i,j].scatter(newTableNorm.iloc[y_km == 3,79], newTableNorm.iloc[y_km ==3,xx],  s=5, c='cyan')
#        ax[i,j].set_title(newTableNorm.columns[xx],loc='center')
#        
#        xx = xx+1
#
#fig.savefig('foo.png')

#affinity propogation
af=AffinityPropagation(preference=-50).fit(newTableNorm)
afClusterInds = af.cluster_centers_indices_
afLabels = af.labels_

testAf=AffinityPropagation(preference=-50).fit(newTestNorm)
testAfClusterInds = testAf.cluster_centers_indices_
testAfLabels = testAf.labels_

#Labels for Classifications
compiledLabels = [nuLabels,y_km,afLabels,newTableTarget]
testLabels = [nuBenchmark,testKLabs,testAfLabels,benchmarkArray]

np.concatenate(compiledLabels)

#Test ANN
modelOne = MLPClassifier(solver='sgd',alpha=1e-5,hidden_layer_sizes=(5,2), random_state=1)
modelOne.fit(newTableFeats,nuLabels)
