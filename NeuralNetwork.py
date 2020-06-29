from sklearn.neural_network import MLPRegressor

import matplotlib.pyplot as plt

import Util as u
import numpy as np
import pandas as pd

def predict(model, covid_data, X_test, column, shown_days):
    print('Predict with neural network...')
        
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

def print_info(model, X_test, y_test, show_loss_plot = True):
    print('metrics for neural network:')        
    print('    Score: ' + str(model.score(X_test, y_test)))
    print('    Loss: ' + str(model.loss_))
    
    if show_loss_plot:
        pd.DataFrame(model.loss_curve_).plot()
        plt.show()
    
def load(unique_name = '', activation = 'relu', solver = 'adam', learning_rate = 'adaptive', random_state = 0, max_iter = 500):
    print('Loading neural network model...')
    
    #https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
    if activation not in ['identity', 'logistic', 'tanh', 'relu']:
        raise Exception('Wrong activation taken. Please change call of method.')
    
    if solver not in ['lbfgs', 'sgd', 'adam']:
        raise Exception('Wrong solver taken. Please change call of method.')
        
    if learning_rate not in ['constant', 'invscaling', 'adaptive']:
        raise Exception('Wrong learning rate taken. Please change call of method.')
        
    if random_state < 0:
        raise Exception('random state can´t be negative.')
        
    if max_iter < 1:
        raise Exception('random state can´t be negative or 0.')
    
    file_name = 'nn_' + unique_name + '_' + activation + '_' + solver + '_' + learning_rate + '_' + str(random_state) + '_' + str(max_iter)
                
    model = u.load_model(file_name)
        
    print('Neural Network loaded.')
    
    return model

def create(X_train, y_train, hidden_layer_sizes = (10, 5, ), unique_name = '', activation = 'relu', solver = 'adam', learning_rate = 'adaptive', random_state = 0, max_iter = 500, save_as_file = False):
    print('Training a new neural network...')
    
    model = MLPRegressor(hidden_layer_sizes = hidden_layer_sizes, activation = activation, solver = solver,
                        learning_rate = learning_rate, random_state = random_state, max_iter = max_iter)
    
    model.fit(X_train, y_train)
    
    file_name = 'nn_' + unique_name + '_' + activation + '_' + solver + '_' + learning_rate + '_' + str(random_state) + '_' + str(max_iter)
    
    if save_as_file:
        u.save_model(model, file_name)
    
    print('Neural Network trained.')
    
    return model