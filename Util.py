#Pascal
import numpy as np
import pickle
import datetime as dt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def prepare_data(covid_data, column, offset, pred_days, filter_by_col = True, scale_data = True):
    print('Preparing data...')
    
    data = covid_data.groupby(['Meldedatum']).sum().sort_values(by = ['Meldedatum'])
    
    if filter_by_col:
        data = data.filter(items=[column])
    
    for i in range(offset):
        i += 1
        temp = data.shift(i, axis = 0)
        data.loc[:,('col-' + str(i))] = temp[column]
        
    temp = data.shift(i - offset, axis = 0)
    data.loc[:,column] = temp
    X = np.array(data.drop([column], 1))
    X_pred = None
    
    if scale_data:
        X = data[offset:]
        X = preprocessing.scale(X)
        X_pred = X[-pred_days:]
        X = X[:-offset:]
    else:
        X = data[offset:-offset:]
    
    y = np.array(data.filter(items=[column]))
    y = y[offset:-offset:]
    
    print('Data prepared.')
    
    return X, X_pred, y, data

def get_train_test(X, y, test_size):
    return train_test_split(X, y, test_size = test_size)

def get_ml_plot_title(shown_days, column):
    if column == 'AnzahlGenesen':
        return 'Anzahl der Genesenen in den letzten ' + str(shown_days) + ' Tagen'
    elif column == 'AnzahlFall':
        return 'Anzahl der Fälle in den letzten ' + str(shown_days) + ' Tagen'
    elif column == 'AnzahlTodesfall':
        return 'Anzahl der Tode in den letzten ' + str(shown_days) + ' Tagen'
    else:
        return None
    
def save_model(model, file_name):
    print('Saving model.')
    with open('Result\\Models\\' + file_name + '.pickle', 'wb') as f:
            pickle.dump(model, f)
            
def load_model(file_name):
    print('Loading model.')
    try:        
        return pickle.load(open('Result\\Models\\' + file_name + '.pickle', 'rb'))
    except (OSError, IOError):
        raise Exception('No file found. Please make sure this model exists or train one.')

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