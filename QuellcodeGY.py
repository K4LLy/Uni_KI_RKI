import keras
import numpy as np
import Util as u
import DataReader as reader

import matplotlib.pyplot as plt

from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from geopy import distance, Nominatim

def predict_nn_with_keras(data, column, print_outputs = False):
    #Pascal Winkler
    print('Predict ' + column + ' with neural network...')
    
    title = u.get_ml_plot_title(80, column)
    if title == None:
        print('Falsche Spalte zum Vorhersagen wurde gewählt: ' + column)
        return
    
    data = data.groupby(['Meldedatum']).sum().sort_values(by = ['Meldedatum'])
    #data = data.filter(items=[column])
    
    offset = 5
    
    for i in range(offset):
        data['col-' + str(i)] = data[column]
        data['col-' + str(i)] = data['col-' + str(i)].shift(i)
    
    #X = preprocessing.scale(data[offset:-offset:])
    X = data[offset:-offset:]
    #X = data
    y = np.array(data[column])[offset:-offset:]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    
    model = Sequential()
    model.add(Dense(6, input_shape = [None, 3 + offset], activation = 'softmax'))
    #model.add(Dense(6, input_shape = [None, 1], activation = 'softmax'))
    model.add(Dense(5, activation = 'relu'))
    model.add(Dense(1, activation = 'softmax'))
    
    if print_outputs:
        model.summary()
        
    model.compile('adam', loss='categorical_crossentropy', metrics = ['accuracy'])
    #model.compile('adam', loss='mean_absolute_error', metrics = ['accuracy'])
    #model.compile('adam', loss='mean_squared_error', metrics = ['accuracy'])
    #model.compile('sgd', loss='mean_absolute_error', metrics = ['accuracy'])
    
    history = model.fit(X_train, y_train, validation_data = (X_test, y_test), batch_size = 2, epochs = 100, verbose = 2)
    
    if print_outputs:
        #plot acc and loss
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss']) 
        plt.title('Model loss') 
        plt.ylabel('Loss') 
        plt.xlabel('Epoch') 
        plt.legend(['Train', 'Test'], loc='upper left') 
        plt.show()
        
def predict_old(data, test_data_count, predicted_days, shown_days, column, save_model = False, print_metric = False, predicted_days_in_future = 0):
    #Lara Ahrens
    #Auslagerung in Funktion mit Verallgemeinerung Pascal Winkler
    print('Predict ' + column + ' with LinReg...')
    data = data.groupby(['Meldedatum']).sum().sort_values(by = ['Meldedatum'])
    data = data.filter(items=[column])
    
    plt_title = ''
    if column == 'AnzahlGenesen':
        plt_title = 'Anzahl der Genesenen in den letzten ' + str(shown_days) + ' Tagen'
    elif column == 'AnzahlFall':
        plt_title = 'Anzahl der Fälle in den letzten ' + str(shown_days) + ' Tagen'
    elif column == 'AnzahlTodesfall':
        plt_title = 'Anzahl der Tode in den letzten ' + str(shown_days) + ' Tagen'
    else:
        print('Falsche Spalte zum Vorhersagen wurde gewählt: ' + column)
        return
    
    newdata, dataframe, y, features_lately, X = prepare_data_lr(data, test_data_count, predicted_days, shown_days, column)
    
    features_train, features_test, labels_train, labels_test = train_test_split(X, y, test_size=0.2)
    linreg = LinearRegression()
    linreg.fit(features_train, labels_train)
    
    if save_model:
        with open('Result\\' + column + '.pickle', 'wb') as f:
            pickle.dump(linreg, f)
    
    labels_pred = linreg.predict(X = features_lately)
    dataframe['Forecast'] = np.nan
    
    for i in range(predicted_days):
        dataframe.iloc[-(predicted_days-i), dataframe.columns.get_loc('Forecast')] = labels_pred[i]
        
    dataframe.loc[dataframe.index[-shown_days]:, column].plot(title=column,  rot=45)
    dataframe.loc[dataframe.index[-shown_days]:, 'Forecast'].plot(title=plt_title, rot=45)
    plt.show()
    print('Predicted ' + column + '.')
    


def prepare_data_lr(dataframe, test_data_count, predicted_days, shown_days, column):
    #Lara Ahrens
    #Auslagerung in Funktion mit Verallgemeinerung Pascal Winkler
    dataframe = dataframe.groupby(['Meldedatum']).sum().sort_values(by = ['Meldedatum'])
    
    for i in range(test_data_count):
        i = i + 1
        new_data = dataframe.shift(i, axis = 0)
        dataframe.loc[:,(column + '-' + str(i))] = new_data[column]
        
    new_data = dataframe.shift(i - test_data_count, axis = 0)
    dataframe.loc[:,column] = new_data  
    #lineares Modell
    features = np.array(dataframe.drop([column], 1))
    
    features = features[test_data_count:]
    features = preprocessing.scale(features)
    labels = np.array(dataframe.filter(items=[column]))
    labels = labels[test_data_count:-test_data_count:]
    features_lately = features[-predicted_days:]
    features = features[:-test_data_count:]
    
    return new_data, dataframe, labels, features_lately, features



def get_closest_station(lat, lon, weather_data):
    #Nina Renken
    closest_dist = -1
    closest_station = None
    
    for row in weather_data.itertuples():
        dist = distance.distance([lat, lon], [row.Geogr_Breite, row.Geogr_Laenge]).km
        if closest_dist == -1 or dist < closest_dist:
            closest_dist = dist
            closest_station = row
    
    return closest_station

print('Reading data')
data_nds_wetter = reader.get_weather_data()
data_covid = reader.get_covid_data()
print('Data read.') 

covid_nds = data_covid[data_covid.Bundesland.eq('Niedersachsen')]

i = 0
percent = 0
naechste_Station = []
for index, row in covid_nds.iterrows():
    station = get_closest_station(row['Landkreis_Lat'], row['Landkreis_Lon'], data_nds_wetter)
    naechste_Station.append(station)
    i += 1
    if ((i / len(covid_nds)) * 100) >= percent:
        print(str(percent) + '% finished')
        percent += 5

covid_nds['naechste_Station'] = naechste_Station
print(covid_nds.head())   
