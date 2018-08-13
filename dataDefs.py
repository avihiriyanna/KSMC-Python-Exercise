#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 00:13:31 2018

@author: avihiriyanna
"""
import numpy as np
import pandas as pd

def labeler(dataList):
    
    uniqueData = dataList.unique()
    
    numUniqueData = len(uniqueData)
    
    labelList = []
    
    initial = 1
    
    for i in range(numUniqueData):
        
        labelList.append([uniqueData[i],initial])
        initial = initial +1
    
    final = pd.DataFrame(labelList)
    
    return final