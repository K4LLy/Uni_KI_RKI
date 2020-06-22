import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

def predict(data, test_data_count, predicted_days, shown_days, column, print_metric = False, predicted_days_in_future = 0):
    print('Predict ' + column + '...')
    dataframe = data.groupby(['Meldedatum']).sum().sort_values(by = ['Meldedatum'])
    dataframe = dataframe.filter(items=[column])
    
    plt_title = ''
    if column == 'AnzahlGenesen':
        plt_title = 'Anzahl der Genesenen in den letzten ' + str(shown_days) + ' Tagen'
    elif column == 'AnzahlFall':
        plt_title = 'Anzahl der Fälle in den letzten ' + str(shown_days) + ' Tagen'
    elif column == 'AnzahlTodesfall':
        plt_title = 'Anzahl der Tode in den letzten ' + str(shown_days) + ' Tagen'
    else:
        print('Falsche Spalte zum Vorhersagen wurde gewählt: ' + column)
        return None
    
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
    
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)
    linreg = LinearRegression()
    linreg.fit(features_train, labels_train)
    labels_pred = linreg.predict(X = features_lately)
    dataframe['Forecast'] = np.nan
    
    for i in range(predicted_days):
        dataframe.iloc[-(predicted_days-i), dataframe.columns.get_loc('Forecast')] = labels_pred[i]
        
    dataframe.loc[dataframe.index[-shown_days]:, column].plot(title=column,  rot=45)
    dataframe.loc[dataframe.index[-shown_days]:, 'Forecast'].plot(title=plt_title, rot=45)
    plt.show()
    print('Predicted ' + column + '.')
    
    if print_metric:
        print('metric for ' + column + ':')
        print('    Score: ' + str(linreg.score(features_test, labels_test))) #Score: andere Metrik, die nicht behandelt wurde. Ähnlich zu MSE
        print('    MSE: ' + str(mean_squared_error(labels[-predicted_days:], labels_pred)))
        print('    MAE: ' + str(mean_absolute_error(labels[-predicted_days:], labels_pred)))
        

def append_future_days(df, days_to_append, column):
    print('Appending future days.')
    indicies = []
    values = []
    
    for i in range(days_to_append):
        date = dt.datetime.strptime(df.tail(1).index.item(), '%Y/%m/%d')
        date = date + dt.timedelta(days = i)
        
        indicies.append(date.strftime('%Y/%m/%d'))
        values.append(np.nan)
        
    vals = {column: values}
    df_future = pd.DataFrame(vals, columns = [column], index = indicies)
    return df.append(df_future)