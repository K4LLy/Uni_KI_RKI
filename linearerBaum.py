# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 15:09:38 2020

@author: AhrensL
"""
import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import DataReader as reader
column = "AnzahlFall"
data = reader.get_covid_data()
dataframe = data.groupby(['Meldedatum']).sum().sort_values(by = ['Meldedatum'])
dataframe = dataframe.filter(items=[column])
test_data_count = 10
predicted_days = 10
shown_days =40
    
plt_title = ''
if column == 'AnzahlGenesen':
    plt_title = 'Anzahl der Genesenen in den letzten ' + str(shown_days) + ' Tagen'
elif column == 'AnzahlFall':
    plt_title = 'Anzahl der Fälle in den letzten ' + str(shown_days) + ' Tagen'
elif column == 'AnzahlTodesfall':
    plt_title = 'Anzahl der Tode in den letzten ' + str(shown_days) + ' Tagen'
else:
    print('Falsche Spalte zum Vorhersagen wurde gewählt: ' + column)

    
for i in range(test_data_count):
    i = i + 1
    newdata = dataframe.shift(i, axis = 0)
    dataframe.loc[:,(column + '-' + str(i))] = newdata[column]
        
newdata = dataframe.shift(i - test_data_count, axis = 0)
dataframe.loc[:,column] = newdata  
    #lineares Modell
features = np.array(dataframe.drop([column], 1))
    
features = features[test_data_count:]
features = preprocessing.scale(features)
labels = np.array(dataframe.filter(items=[column]))
labels = labels[test_data_count:-test_data_count:]
features_lately = features[-predicted_days:]
features = features[:-test_data_count:]
    
from sklearn import tree


reg_tree = tree.DecisionTreeRegressor()
reg_tree = reg_tree.fit(features, labels)

#tree.plot_tree(reg_tree)
labels_pred = reg_tree.predict(X = features_lately)
dataframe['Forecast'] = np.nan
    
for i in range(predicted_days):
    dataframe.iloc[-(predicted_days-i), dataframe.columns.get_loc('Forecast')] = labels_pred[i]
        
dataframe.loc[dataframe.index[-shown_days]:, column].plot(title=column,  rot=45)
dataframe.loc[dataframe.index[-shown_days]:, 'Forecast'].plot(title=plt_title, rot=45)
plt.show()
print('Predicted ' + column + '.')
    

print('metric for ' + column + ':')
print('    MSE: ' + str(mean_squared_error(labels[-predicted_days:], labels_pred)))
print('    MAE: ' + str(mean_absolute_error(labels[-predicted_days:], labels_pred)))
#
#X = [[0, 0], [1, 1]]
#Y = [0, 1]
#clf = tree.DecisionTreeRegressor()
#clf = clf.fit(X, Y)
#y_pred = clf.predict([[2., 2.]])
#print(str(y_pred))


#y_predPro = clf.predict_proba([[2., 2.]])
#print(str(y_predPro))


#from sklearn.datasets import load_iris
#from sklearn import tree
#X, y = load_iris(return_X_y=True)
#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(X, y)

#tree.plot_tree(clf) 