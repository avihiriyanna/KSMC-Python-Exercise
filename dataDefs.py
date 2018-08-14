#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 00:13:31 2018

@author: avihiriyanna
"""
import numpy as np
import pandas as pd

def quartileLabeler(dataList):

    nuLabels =[]
    
    for i in range(len(dataList)):
    
        if dataList.iloc[i]<=129975:
            nuLabels.append(0)
        elif dataList.iloc[i]>129975 and dataList.iloc[i]<=163000:
            nuLabels.append(1)
        elif dataList.iloc[i]>163000 and dataList.iloc[i]<=214000:
            nuLabels.append(2)
        elif dataList.iloc[i]>214000:
            nuLabels.append(3)
    
    return nuLabels