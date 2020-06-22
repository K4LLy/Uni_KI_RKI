# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 16:19:45 2020

@author: AhrensL
"""


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

predicted_days = 1
shown_days =10
    
plt_title = ''
if column == 'AnzahlGenesen':
    plt_title = 'Anzahl der Genesenen in den letzten ' + str(shown_days) + ' Tagen'
elif column == 'AnzahlFall':
    plt_title = 'Anzahl der Fälle in den letzten ' + str(shown_days) + ' Tagen'
elif column == 'AnzahlTodesfall':
    plt_title = 'Anzahl der Tode in den letzten ' + str(shown_days) + ' Tagen'
else:
    print('Falsche Spalte zum Vorhersagen wurde gewählt: ' + column)

    

newdata = dataframe.shift(1, axis = 0)
dataframe.loc[:,('Features')] = newdata[column]
        


    #lineares Modell
features = np.array(dataframe.drop([column], 1))
    
features = features[1:]

labels = np.array(dataframe.filter(items=[column]))
labels = labels[1:-1:]
features_lately = features[-1:]
features = features[:-1:]
    
from sklearn import tree


reg_tree = tree.DecisionTreeRegressor()
reg_tree = reg_tree.fit(features, labels)

#tree.plot_tree(reg_tree)
labels_pred = reg_tree.predict(X = features_lately)
dataframe['Forecast'] = np.nan
    

print('Vorhergesagter Anzahl Fall für den nächsten Tag in der Zukunft: ' + str(labels_pred[0]))
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