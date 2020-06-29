# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 12:00:02 2020

@author: AhrensL
"""

#zusammenbauen der Daten
import DataReader as reader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
#holen der Daten aus dem Reader
covid_data = reader.get_covid_data()


covid_data = covid_data.groupby(['Kalenderwoche', 'Bundesland']).sum()
kalenderwoche = np.array(covid_data.index.get_level_values(0))
bundesland = np.array(covid_data.index.get_level_values(1))
covid_data.loc[:,('KW')] = kalenderwoche
covid_data.loc[:,('Bundesland')] = bundesland


values = np.array(covid_data.filter(items=['Bundesland']))
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
# define example


# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded) 

covid_data.loc[:,('Baden-Wüttemberg')] = onehot_encoded[ :, 0]
covid_data.loc[:,('Bayern')] = onehot_encoded[ :, 1]
covid_data.loc[:,('Berlin')] = onehot_encoded[ :, 2]
covid_data.loc[:,('Brandenburg')] = onehot_encoded[ :, 3]
covid_data.loc[:,('Bremen')] = onehot_encoded[ :, 4]
covid_data.loc[:,('Hamburg')] = onehot_encoded[ :, 5]
covid_data.loc[:,('Hessen')] = onehot_encoded[ :, 6]
covid_data.loc[:,('Mecklenburg-Vorpommern')] = onehot_encoded[ :, 7]
covid_data.loc[:,('Niedersachsen')] = onehot_encoded[ :, 8]
covid_data.loc[:,('Nordrhein-Westfalen')] = onehot_encoded[ :, 9]
covid_data.loc[:,('Rheinland-Pfalz')] = onehot_encoded[ :, 10]
covid_data.loc[:,('Saarland')] = onehot_encoded[ :, 11]
covid_data.loc[:,('Sachsen')] = onehot_encoded[ :, 12]
covid_data.loc[:,('Sachsen-Anhalt')] = onehot_encoded[ :, 13]
covid_data.loc[:,('Schleswig-Holstein')] = onehot_encoded[ :, 14]
covid_data.loc[:,('Thüringen')] = onehot_encoded[ :, 15]


#ab kalenderwoche 12 werden Maßnahmen getroffen

massnahmen = []
kalenderwochen_nr = []
for indexes, row in covid_data.iterrows():
    kalenderwoche = row['KW']
    kalenderwoche_nr = int(kalenderwoche)
    kalenderwochen_nr.append(kalenderwoche_nr)
    if kalenderwoche_nr >= 12:
        massnahmen.append(1)
    else:
        massnahmen.append(0)
        
covid_data.loc[:,('MassnahmenJN')] = massnahmen
covid_data.loc[:,('Kalenderwoche')] = kalenderwochen_nr

#labels aufbauen: Daten der nächsten Woche als Feature aufbauen
#features: Kalenderwoche, Bundesland (....), massnahmen
#label: anzahl fall 
dataframe_der_features = covid_data.filter(items= ['Kalenderwoche', 'MassnahmenJN', 'Baden-Wüttemberg', 'Bayern', 'Berlin', 'Brandenburg','Bremen', 'Hamburg', 'Hessen', 'Mecklenburg-Vorpommern','Niedersachsen', 'Nordrhein-Westfalen', 'Rheinland-Pfalz', 'Saarland', 'Sachsen', 'Sachsen-Anhalt', 'Schleswig-Holstein', 'Thüringen' ])
features = np.array(dataframe_der_features)
#features = preprocessing.scale(features)
dataframe_der_labels = covid_data.filter(items = ['AnzahlFall'])
labels = np.array(dataframe_der_labels)
#Test und Trainingssatz

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)
#lineare Regression zusammenbauen und trainieren
linreg = LinearRegression()
linreg.fit(features_train, labels_train)

    
#feature zusammenbauen: Anzahl Fall für jedes Bundesland in der KW 23 mit Massnahmen

feature_to_predict = np.array([[23, 1, 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #Baden-Wüttemberg
                               [23, 1, 0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #Bayern
                               [23, 1, 0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0], #Berlin
                               [23, 1, 0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0], #Brandenburg
                               [23, 1, 0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], #Bremen
                               [23, 1, 0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0], #Hamburg
                               [23, 1, 0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0], #Hessen
                               [23, 1, 0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0], #Mecklenburg-Vorpommern
                               [23, 1, 0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0], #Niedersachsen
                               [23, 1, 0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0], #Nordrhein-Westfalen
                               [23, 1, 0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0], #Rheinland-Pfalz
                               [23, 1, 0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0], #Saarland 
                               [23, 1, 0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0], #Sachsen
                               [23, 1, 0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0], #Sachsen-Anhalt
                               [23, 1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0], #Schleswig-Holstein
                               [23, 1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]]) #Thüringen
feature_fuer_fehler= np.array([ [21, 1, 0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]]) #Index 219 aus der Covid-Data
labels_pred = linreg.predict(X = feature_to_predict)
label_fuer_fehler = linreg.predict(X=feature_fuer_fehler)
print('Lineare Regression: Fälle in KW 23:' + '\n' +
      'Fälle in Baden-Wüttemberg: ' + str(labels_pred[0])+'\n' +
      'Fälle in Bayern: ' + str(labels_pred[1])+'\n' +
      'Fälle in Berlin: ' + str(labels_pred[2])+'\n' +
      'Fälle in Brandenburg: ' + str(labels_pred[3])+'\n' +
      'Fälle in Bremen: ' + str(labels_pred[4])+'\n' +
      'Fälle in Hamburg: ' + str(labels_pred[5])+'\n' +
      'Fälle in Hessen: ' + str(labels_pred[6])+'\n' +
      'Fälle in Mecklenburg-Vorpommern: ' + str(labels_pred[7])+'\n' +
      'Fälle in Niedersachsen: ' + str(labels_pred[8])+'\n' +
      'Fälle in Nordrhein-Westfahlen: ' + str(labels_pred[9])+'\n' +
      'Fälle in Rheinland-Pfalz: ' + str(labels_pred[10])+'\n' +
      'Fälle in Saarland: ' + str(labels_pred[11])+'\n' +
      'Fälle in Sachsen: ' + str(labels_pred[12])+'\n' +
      'Fälle in Sachsen-Anhalt: ' + str(labels_pred[13])+'\n' +
      'Fälle in Schleswig-Holstein: ' + str(labels_pred[14])+'\n' +
      'Fälle in Thüringen: ' + str(labels_pred[15])
      )
print('Metrik: ')
print('    Score: ' + str(linreg.score(features_test, labels_test))) #Score: andere Metrik, die nicht behandelt wurde. Ähnlich zu MSE
print('    MSE: ' + str(mean_squared_error(labels[219], label_fuer_fehler)))
print('    MAE: ' + str(mean_absolute_error(labels[219], label_fuer_fehler)))
reg_tree = tree.DecisionTreeRegressor()
reg_tree = reg_tree.fit(features, labels)

#tree.plot_tree(reg_tree)
labels_pred_tree = reg_tree.predict(X = feature_to_predict)
label_fuer_fehler_tree = reg_tree.predict(X=feature_fuer_fehler)
print('Baum: : Fälle in KW 23:' + '\n' +
      'Fälle in Baden-Wüttemberg: ' + str(labels_pred_tree[0])+'\n' +
      'Fälle in Bayern: ' + str(labels_pred_tree[1])+'\n' +
      'Fälle in Berlin: ' + str(labels_pred_tree[2])+'\n' +
      'Fälle in Brandenburg: ' + str(labels_pred_tree[3])+'\n' +
      'Fälle in Bremen: ' + str(labels_pred_tree[4])+'\n' +
      'Fälle in Hamburg: ' + str(labels_pred_tree[5])+'\n' +
      'Fälle in Hessen: ' + str(labels_pred_tree[6])+'\n' +
      'Fälle in Mecklenburg-Vorpommern: ' + str(labels_pred_tree[7])+'\n' +
      'Fälle in Niedersachsen: ' + str(labels_pred_tree[8])+'\n' +
      'Fälle in Nordrhein-Westfahlen: ' + str(labels_pred_tree[9])+'\n' +
      'Fälle in Rheinland-Pfalz: ' + str(labels_pred_tree[10])+'\n' +
      'Fälle in Saarland: ' + str(labels_pred_tree[11])+'\n' +
      'Fälle in Sachsen: ' + str(labels_pred_tree[12])+'\n' +
      'Fälle in Sachsen-Anhalt: ' + str(labels_pred_tree[13])+'\n' +
      'Fälle in Schleswig-Holstein: ' + str(labels_pred_tree[14])+'\n' +
      'Fälle in Thüringen: ' + str(labels_pred_tree[15]))
print('Metrik: ')
print('    MSE: ' + str(mean_squared_error(labels[219], label_fuer_fehler_tree)))
print('    MAE: ' + str(mean_absolute_error(labels[219], label_fuer_fehler_tree)))

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression

#Labelvorhersage wird größer bzw kleiner, wenn HiddenLayer größer / kleiner
#Activation sagt geringe Werte voraus
#plotten der Loss-Kurve funktioniert nur beim Default Optimizer
mlp_regr = MLPRegressor(activation='relu',random_state=1, max_iter=500, solver ='lbfgs', learning_rate='invscaling').fit(features_train, labels_train)
mlp_labels_predict = mlp_regr.predict(X = feature_to_predict)
test_prediction = mlp_regr.predict(X = features_test)
label_fuer_fehler_knn = mlp_regr.predict(X=feature_fuer_fehler)
print('Neuronales Netz : Fälle in KW 23:' + '\n' +
      'Fälle in Baden-Wüttemberg: ' + str(mlp_labels_predict[0])+'\n' +
      'Fälle in Bayern: ' + str(mlp_labels_predict[1])+'\n' +
      'Fälle in Berlin: ' + str(mlp_labels_predict[2])+'\n' +
      'Fälle in Brandenburg: ' + str(mlp_labels_predict[3])+'\n' +
      'Fälle in Bremen: ' + str(mlp_labels_predict[4])+'\n' +
      'Fälle in Hamburg: ' + str(mlp_labels_predict[5])+'\n' +
      'Fälle in Hessen: ' + str(mlp_labels_predict[6])+'\n' +
      'Fälle in Mecklenburg-Vorpommern: ' + str(mlp_labels_predict[7])+'\n' +
      'Fälle in Niedersachsen: ' + str(mlp_labels_predict[8])+'\n' +
      'Fälle in Nordrhein-Westfahlen: ' + str(mlp_labels_predict[9])+'\n' +
      'Fälle in Rheinland-Pfalz: ' + str(mlp_labels_predict[10])+'\n' +
      'Fälle in Saarland: ' + str(mlp_labels_predict[11])+'\n' +
      'Fälle in Sachsen: ' + str(mlp_labels_predict[12])+'\n' +
      'Fälle in Sachsen-Anhalt: ' + str(mlp_labels_predict[13])+'\n' +
      'Fälle in Schleswig-Holstein: ' + str(mlp_labels_predict[14])+'\n' +
      'Fälle in Thüringen: ' + str(mlp_labels_predict[15]))
print('Metrik: ')
print('Score: '+ str(mlp_regr.score(features_test, labels_test)))
print('Loss: '+str(mlp_regr.loss_))
kalenderwoche_to_plot = features_test[:,0]
plt.scatter(kalenderwoche_to_plot, labels_test, color= 'blue')
plt.scatter(kalenderwoche_to_plot, test_prediction, color = 'red')
plt.ylabel('Anzahl der Fälle')
plt.xlabel('Kalenderwoche')
plt.title('Tatsächliche und vorausgesagte Fälle neuronales Netz (Blau: Tatsächlich, Rot: Vorausgesagt)')
plt.show()

