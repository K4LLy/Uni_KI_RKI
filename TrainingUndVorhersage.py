import numpy as np
import DataReader as reader
covid_data = reader.get_covid_data()
dataframe = covid_data.groupby(['Meldedatum']).sum()

dataframe = dataframe.filter(items=['AnzahlGenesen'])

for i in range(5):
    i = i +1
    newdata = dataframe.shift(i, axis = 0)
    dataframe.loc[:,('Genes-'+str(i))] = newdata['AnzahlGenesen']
    
newdata = dataframe.shift(i-5, axis = 0)
dataframe.loc[:,'AnzahlGenesen'] = newdata  

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#lineares Modell
features = np.array(dataframe.drop(['AnzahlGenesen'],1))

features = features[5:]
features = preprocessing.scale(features)
labels = np.array(dataframe.filter(items=['AnzahlGenesen']))
labels = labels[5:-5:]
features_lately = features[-5:]
features = features[:-5:]


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)
linear_classifier = LinearRegression()
linear_classifier.fit(features_train, labels_train)
score = linear_classifier.score(features_test,labels_test)
print(score)
labels_pred = linear_classifier.predict(X=features_lately)
print(labels_pred)
dataframe['Forecast'] = np.nan
for i in range(5):
    dataframe.iloc[-(5-i), dataframe.columns.get_loc('Forecast')] = labels_pred[i]
    
dataframe.loc[dataframe.index[-30]:, 'AnzahlGenesen'].plot(title="AnzahlGenesen",  rot=45)
dataframe.loc[dataframe.index[-30]:, 'Forecast'].plot(title="AnzahlGenesen in den letzten 30 Tagen: Tatsächlich und Vorhersage", rot=45)
