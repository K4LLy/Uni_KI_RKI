# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:05:02 2020

@author: RenkenN
"""


import matplotlib.pyplot as plt
import numpy as np
import DataReader as reader
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

dataframe = reader.getCovidData()

print(dataframe.dtypes)

dataframe.loc[:,'Meldedatum'] = pd.to_datetime(dataframe.Meldedatum, format='%b %d, %Y')

print(dataframe.dtypes)