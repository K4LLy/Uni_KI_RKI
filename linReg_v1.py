import matplotlib.pyplot as plt
import numpy as np
import DataReader as reader
import pandas as pd
from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Daten holen und vorbereiten - Meldedatum und AnzahlTodesfall rausfiltern
covid_data = reader.get_covid_data()
dataframe = covid_data.groupby(['Meldedatum']).sum()

#dataframe.loc[:,'Meldedatum'] = pd.to_datetime(dataframe.Meldedatum, format='%Y/%m/%d')
#dataframe.set_index('Meldedatum', inplace=True)
dataframe = dataframe.filter(items=['AnzahlTodesfall'])

# Nach Meldedatum sortieren und aufsummieren
#dataframe = dataframe.sort_values(by=['Meldedatum'])
#dataframe = dataframe.cumsum()

# hierdurch werden 5 weitere Spalten angelegt mit den Daten für jeweils 6-10 Tage vor dem tatsächlichen Meldedatum
# mit den 5 weiteren Spalten wird trainiert, um den tatsächlichen Fall, also die letzten 5 Tage zu prognostizieren
# mit der Spalte 'AnzahlTodesfall' wird die Prognose dann verglichen, sie ist somit das Label
for i in range(5):
    i = i + 1
    newdata = dataframe.shift(i, axis = 0)
    dataframe.loc[:, ('AnzahlTodesfall-' + str(i))] = newdata

newdata = dataframe.shift(i-5, axis = 0)
dataframe.loc[:, 'AnzahlTodesfall'] = newdata


features = np.array(dataframe.drop(['AnzahlTodesfall'], 1))
features = features[5:]

# alle features bekommen den selbsen Wertebereich, Standardabweichung = 1
features = preprocessing.scale(features)

features_lately = features[-5:]
features = features[:-5:]

print("features[-50:]", features[:-5:])

labels = np.array(dataframe.filter(items=['AnzahlTodesfall']))
labels = labels[5:-5:]

print("len(dataframe): ", len(dataframe))
print("len(features): ", len(features))
print("len(features_lately): ", len(features_lately))
print("len(labels): ", len(labels))

# Datenaufteilung in 20% Testdaten (test_size=0.2) und Trainingsdaten mit der Funktino "train_test_split"
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

linear_classifier = LinearRegression()
linear_classifier.fit(features_train, labels_train)

# Score = 1 wäre ideal/perfekt
score = linear_classifier.score(features_test, labels_test)
print("Score: ", score)

# Berechnung der neuen Label mit sklearn
labels_pred = linear_classifier.predict(X=features_lately)
print("Vorhersage: ", labels_pred)

dataframe['Prognose'] = np.nan
for i in range(5):
    dataframe.iloc[-(5-i), dataframe.columns.get_loc('Prognose')] = labels_pred[i]
    
dataframe.loc[dataframe.index[-30]:, 'AnzahlTodesfall'].plot()
dataframe.loc[dataframe.index[-30]:, 'Forecast'].plot()