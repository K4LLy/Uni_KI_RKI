# -*- coding: utf-8 -*-

import DataReader as reader
import numpy as np
covid_data = reader.get_covid_data()
#geschlechter = np.array(covid_data.filter(items=['Geschlecht']))

covid_data = covid_data.groupby(['Meldedatum', 'Geschlecht', 'Altersgruppe']).sum()
geschlechter = np.array(covid_data.index.get_level_values(1))
altersgruppe = np.array(covid_data.index.get_level_values(2))
covid_data.loc[:,('Geschlecht2')] = geschlechter
covid_data.loc[:,('Altersgruppe2')] = altersgruppe

#covid_data.reset_index(level=0, inplace=True)
values = np.array(covid_data.filter(items=['Geschlecht2']))
values_alter = np.array(covid_data.filter(items=['Altersgruppe2']))
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
# define example


# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
integer_encoded_alter = label_encoder.fit_transform(values_alter)

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
integer_encoded_alter = integer_encoded_alter.reshape(len(integer_encoded_alter), 1)
onehot_encoded_alter = onehot_encoder.fit_transform(integer_encoded_alter)
 #0 = männlich, 1 = weiblich, 2 = unbekannt
covid_data.loc[:,('maennlich')] = onehot_encoded[ :, 0]
covid_data.loc[:,('weiblich')] = onehot_encoded[ :, 1]
covid_data.loc[:,('unbekannt')] = onehot_encoded[ :, 2]

covid_data.loc[:,('A00-A04')] = onehot_encoded_alter[ :, 0]
covid_data.loc[:,('A05-A14')] = onehot_encoded_alter[ :, 1]
covid_data.loc[:,('A15-A34')] = onehot_encoded_alter[ :, 2]
covid_data.loc[:,('A35-A59')] = onehot_encoded_alter[ :, 3]
covid_data.loc[:,('A60-A79')] = onehot_encoded_alter[ :, 4]
covid_data.loc[:,('A80+')] = onehot_encoded_alter[ :, 5]
covid_data.loc[:,('A_unbekannt')] = onehot_encoded_alter[ :, 6]

covid_data=covid_data.drop(['Geschlecht2'], 1)
covid_data=covid_data.drop(['Altersgruppe2'], 1)
covid_data = covid_data.drop(['AnzahlTodesfall'], 1)
covid_data = covid_data.drop(['AnzahlGenesen'], 1)


features = np.array(covid_data.drop(['AnzahlFall'], 1))
    

features = preprocessing.scale(features)
labels = np.array(covid_data.filter(items=['AnzahlFall']))

 
from sklearn import tree
import matplotlib.pyplot as plt

reg_tree = tree.DecisionTreeRegressor()
fig, ax = plt.subplots(figsize=(27,15))
tree.plot_tree(reg_tree.fit(features, labels), max_depth=10, fontsize=10)

fig.savefig("Result\\" + 'Linearer_Baum_Fälle_Geschlecht_Altersgruppe' + ".png")
plt.show()

featureToPredict = np.array([[1,0,0, 1,0,0,0,0,0,0],#Männlich und Altersgruppe
                             [1,0,0, 0,1,0,0,0,0,0],
                             [1,0,0, 0,0,1,0,0,0,0],
                             [1,0,0, 0,0,0,1,0,0,0],
                             [1,0,0, 0,0,0,0,1,0,0],
                             [1,0,0, 0,0,0,0,0,1,0]])
labels_pred = reg_tree.predict(X = featureToPredict)
print('Vorhergesagte Anzahl Fall pro Tag : \n'+
      'für männlich: \n'+
      'für Altersgruppe 00-04 ' + str(labels_pred[0])+ '\n' +
      'für Altersgruppe 05-14 ' + str(labels_pred[1])+'\n' +
      'für Altersgruppe 15-34 ' + str(labels_pred[2])+'\n' +
      'für Altersgruppe 35-59 ' + str(labels_pred[3])+'\n' +
      'für Altersgruppe 60-79 ' + str(labels_pred[4])+'\n' +
      'für Altersgruppe 80+ ' + str(labels_pred[5]))
# invert first example
#inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
#print(inverted)