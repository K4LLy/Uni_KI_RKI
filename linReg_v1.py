import matplotlib.pyplot as plt
import numpy as np
import DataReader as reader
import pandas as pd
from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Daten holen und vorbereiten - Meldedatum und AnzahlTodesfall rausfiltern
dataframe = reader.getCovidData()

dataframe.loc[:,'Meldedatum'] = pd.to_datetime(dataframe.Meldedatum, format='%Y/%m/%d')
dataframe.set_index('Meldedatum', inplace=True)
dataframe = dataframe.filter(items=['Meldedatum', 'AnzahlTodesfall'])

#Nach Meldedatum sortieren und aufsummieren
dataframe = dataframe.sort_values(by=['Meldedatum'])
dataframe = dataframe.cumsum()

dataframe.plot()

#print(dataframe.tail(20))

features = np.array(dataframe.filter(items=['AnzahlTodesfall']))
labels = np.array(dataframe.filter(items=['AnzahlTodesfall']))

#alle features bekommen den selbsen Wertebereich
features = preprocessing.scale(features)

#Datenaufteilung in 20% Testdaten (test_size=0.2) und Trainingsdaten mit der Funktino "train_test_split"
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

linear_classifier = LinearRegression()
linear_classifier.fit(features_train, labels_train)

score = linear_classifier.score(features_test, labels_test)
print(score)