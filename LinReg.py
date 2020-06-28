import numpy as np
import matplotlib.pyplot as plt
import Util as u
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
        
def predict(model, covid_data, X_test, column, shown_days):
    print('Predict data...')
        
    title = u.get_ml_plot_title(shown_days, column)
    if title == None:
        raise Exception('Falsche Spalte zum Vorhersagen wurde gewählt: ' + column)
        
    y_pred = model.predict(X_test)
    
    covid_data['Forecast'] = np.nan
    
    for i in range(len(y_pred)):
        covid_data.iloc[-(len(y_pred) - i), covid_data.columns.get_loc('Forecast')] = y_pred[i]
        
    covid_data.loc[covid_data.index[-shown_days]:, column].plot(title=title,  rot=45)
    covid_data.loc[covid_data.index[-shown_days]:, 'Forecast'].plot(title=title, rot=45)
    plt.show()
    
    print('Prediction done.')
    
    return y_pred
    
def create(X_train, y_train, unique_name = '', save_as_file = False):
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    file_name = 'LinReg_' + unique_name
    
    if save_as_file:
        u.save_model(model, file_name)
        
    return model

def load(unique_name = ''):
    print('Loading LinReg model...')
        
    file_name = 'LinReg_' + unique_name                
    model = u.load_model(file_name)
        
    print('LinReg loaded.')
    
    return model
    
def print_info(model, X_test, y_test, y_pred):
    print('metrics for LinReg:')
    print('    Score: ' + str(model.score(X_test, y_test))) #Score: andere Metrik, die nicht behandelt wurde. Ähnlich zu MSE
    print('    MSE: ' + str(mean_squared_error(y_test, y_pred)))
    print('    MAE: ' + str(mean_absolute_error(y_test, y_pred)))