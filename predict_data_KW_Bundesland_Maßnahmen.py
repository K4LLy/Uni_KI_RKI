

#import
import DataReader as reader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
from sklearn.neural_network import MLPRegressor


def prepareData(covid_data, bundesland_to_filter):
    
    covid_data = covid_data.groupby(['Kalenderwoche', 'Bundesland']).sum()
    kalenderwoche = np.array(covid_data.index.get_level_values(0))
    bundesland = np.array(covid_data.index.get_level_values(1))
    covid_data.loc[:,('KW')] = kalenderwoche
    covid_data.loc[:,('Bundesland')] = bundesland


    values = np.array(covid_data.filter(items=['Bundesland']))


# integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)


    covid_data.loc[:,('Baden-Württemberg')] = onehot_encoded[ :, 0]
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
#r-0-Faktor?
    r_null_faktor = []
    massnahmen = []
    kalenderwochen_nr = []
    for indexes, row in covid_data.iterrows():
        kalenderwoche = row['KW']
        anzahlFall = row ['AnzahlFall']
        anzahlGenesen = row['AnzahlGenesen']
        if anzahlGenesen != 0:
            r_null = anzahlFall / anzahlGenesen
            r_null_faktor.append(r_null)
        else:
            r_null_faktor.append(anzahlFall)
        kalenderwoche_nr = int(kalenderwoche)
        kalenderwochen_nr.append(kalenderwoche_nr)
        if kalenderwoche_nr >= 12:
            massnahmen.append(1)
        else:
            massnahmen.append(0)
        
    covid_data.loc[:,('MassnahmenJN')] = massnahmen
    covid_data.loc[:,('Kalenderwoche')] = kalenderwochen_nr
    covid_data.loc[:,('R_Null_Faktor')] = r_null_faktor
    covid_data = covid_data.loc[covid_data[bundesland_to_filter] == 1] #Filter nach gefragtem Bundesland
    return covid_data

def printData(labels_pred, str_to_predict, regression, kalenderwoche, bundesland):
    print(regression +' '+str_to_predict + ' in KW '+str(kalenderwoche) +' :' + '\n' +
          str_to_predict + ' in '  +bundesland  + ': '   + str(labels_pred[0])
          )
    
def predictData(covid_data, kalenderwoche, column_to_predict, massnahmenJN, bundesland):
   
    #label: anzahl fall 
    dataframe_der_features = covid_data.filter(items= ['Kalenderwoche', 'MassnahmenJN' ])
    features = np.array(dataframe_der_features)
    #features = preprocessing.scale(features)
    dataframe_der_labels = covid_data.filter(items = [column_to_predict])
    str_to_predict = ''
    if column_to_predict == 'AnzahlFall':
       str_to_predict = 'Anzahl der Fälle' 
    elif column_to_predict =='AnzahlGenesen':
        str_to_predict = 'Gesunde Fälle'
    elif column_to_predict =='AnzahlTodesfall':
        str_to_predict = 'Tote Fälle'
    elif column_to_predict == 'R_Null_Faktor':
        str_to_predict = 'R0-Faktor'
    labels = np.array(dataframe_der_labels)
    #Test und Trainingssatz

    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)
    #lineare Regression zusammenbauen und trainieren
    linreg = LinearRegression()
    linreg.fit(features_train, labels_train)

    
    #feature zusammenbauen: Anzahl Fall in der KW mit/ohne Massnahmen

    feature_to_predict = np.array([[kalenderwoche, massnahmenJN]])
    
    #Lineare Regression
    labels_pred = linreg.predict(X = feature_to_predict)
    label_fuer_fehler_linReg = linreg.predict(X=features_test)
    printData(labels_pred, str_to_predict, 'Lineare Regression', kalenderwoche, bundesland)
    print('Metrik der linearen Regression: ')
    print('    Score: ' + str(linreg.score(features_test, labels_test))) #Score: andere Metrik, die nicht behandelt wurde. Ähnlich zu MSE
    print('    MSE: ' + str(mean_squared_error(labels_test, label_fuer_fehler_linReg)))
    print('    MAE: ' + str(mean_absolute_error(labels_test, label_fuer_fehler_linReg)))
    #Baum
    reg_tree = tree.DecisionTreeRegressor()
    reg_tree = reg_tree.fit(features_train, labels_train)

    labels_pred_tree = reg_tree.predict(X = feature_to_predict)
    label_fuer_fehler_tree = reg_tree.predict(X=features_test)
    printData(labels_pred_tree, str_to_predict, 'Baum', kalenderwoche, bundesland)
    print('Metrik Baum: ')
    print('    Score: ' + str(reg_tree.score(features_test, labels_test))) 
    print('    MSE: ' + str(mean_squared_error(labels_test, label_fuer_fehler_tree)))
    print('    MAE: ' + str(mean_absolute_error(labels_test, label_fuer_fehler_tree)))

    #neuronales Netz
    
    mlp_regr = MLPRegressor(activation='relu',random_state=1, max_iter=500, solver ='lbfgs', learning_rate='invscaling').fit(features_train, labels_train)
    
    mlp_labels_predict = mlp_regr.predict(X = feature_to_predict)
    test_prediction = mlp_regr.predict(X= features_test)
    printData(mlp_labels_predict, str_to_predict, 'Neuronales Netz', kalenderwoche, bundesland)
    print('Metrik neuronales Netz: ')
    print('    Score: '+ str(mlp_regr.score(features_test, labels_test)))
    print('    Loss: '+str(mlp_regr.loss_))
    #pd.DataFrame(mlp_regr.loss_curve_).plot()
    kalenderwoche_to_plot = features_test[:,0]
    plt.scatter(kalenderwoche_to_plot, labels_test, color= 'blue')
    plt.scatter(kalenderwoche_to_plot, test_prediction, color = 'red')
    plt.ylabel('Anzahl der Fälle')
    plt.xlabel('Kalenderwoche')
    plt.title('Tatsächliche und vorausgesagte Fälle neuronales Netz ' +bundesland + '(Blau: Tatsächlich, Rot: Vorausgesagt)')
    plt.show()
   
def predict_Data(covid_data, kalenderwoche, data_to_predict, massnahmenJN, bundesland):
    dataframe = prepareData(covid_data, bundesland)
    predictData(dataframe, kalenderwoche, data_to_predict, massnahmenJN, bundesland)

    
covid_data = reader.get_covid_data()
bundesland = 'Nordrhein-Westfalen'
test = predict_Data(covid_data, 23, 'AnzahlFall', 1, bundesland) #AnzahlFall, AnzahlTodesfall, AnzahlGenesen, R_Null_Faktor



#bundesland = 'Bayern'
#predict_Data(covid_data, 23, 'AnzahlFall', 1, bundesland) #AnzahlFall, AnzahlTodesfall, AnzahlGenesen, R_Null_Faktor
#bundesland = 'Mecklenburg-Vorpommern'
#predict_Data(covid_data, 23, 'AnzahlFall', 1, bundesland) #AnzahlFall, AnzahlTodesfall, AnzahlGenesen, R_Null_Faktor
