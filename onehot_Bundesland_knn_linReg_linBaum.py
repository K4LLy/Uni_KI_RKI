#Erster Entwurf für die Vorraussage der Fälle/Todesfälle/Gesunde pro Bundesland anhand von Onehot-Encoding 
#import
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
from sklearn.neural_network import MLPRegressor


def prepareData(covid_data):
    
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


    covid_data.loc[:,('Baden-Wüttemberg')] = onehot_encoded[ :, 0]
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


    #ab kalenderwoche 12 werden Maßnahmen getroffen, hier sehr verallgemeinert

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
    return covid_data

def printData(labels_pred, str_to_predict, regression, kalenderwoche):
    print(regression +' '+str_to_predict + ' in KW '+str(kalenderwoche) +' :' + '\n' +
          str_to_predict + ' in Baden-Wüttemberg: '       + str(labels_pred[0])+'\n' +
          str_to_predict + ' in Bayern: '                 + str(labels_pred[1])+'\n' +
          str_to_predict + ' in Berlin: '                 + str(labels_pred[2])+'\n' +
          str_to_predict + ' in Brandenburg: '            + str(labels_pred[3])+'\n' +
          str_to_predict + ' in Bremen: '                 + str(labels_pred[4])+'\n' +
          str_to_predict + ' in Hamburg: '                + str(labels_pred[5])+'\n' +
          str_to_predict + ' in Hessen: '                 + str(labels_pred[6])+'\n' +
          str_to_predict + ' in Mecklenburg-Vorpommern: ' + str(labels_pred[7])+'\n' +
          str_to_predict + ' in Niedersachsen: '          + str(labels_pred[8])+'\n' +
          str_to_predict + ' in Nordrhein-Westfahlen: '   + str(labels_pred[9])+'\n' +
          str_to_predict + ' in Rheinland-Pfalz: '        + str(labels_pred[10])+'\n' +
          str_to_predict + ' in Saarland: '               + str(labels_pred[11])+'\n' +
          str_to_predict + ' in Sachsen: '                + str(labels_pred[12])+'\n' +
          str_to_predict + ' in Sachsen-Anhalt: '         + str(labels_pred[13])+'\n' +
          str_to_predict + ' in Schleswig-Holstein: '     + str(labels_pred[14])+'\n' +
          str_to_predict + ' in Thüringen: '              + str(labels_pred[15])
          )
    
def predictData(covid_data, kalenderwoche, column_to_predict, massnahmenJN):
    #labels aufbauen: Daten der nächsten Woche als Feature aufbauen
    #features: Kalenderwoche, Bundesland (dargestellt als 0 und 1), massnahmen
    #label: anzahl fall 
    dataframe_der_features = covid_data.filter(items= ['Kalenderwoche', 'MassnahmenJN', 'Baden-Wüttemberg', 'Bayern', 'Berlin', 'Brandenburg','Bremen', 'Hamburg', 'Hessen', 'Mecklenburg-Vorpommern','Niedersachsen', 'Nordrhein-Westfalen', 'Rheinland-Pfalz', 'Saarland', 'Sachsen', 'Sachsen-Anhalt', 'Schleswig-Holstein', 'Thüringen' ])
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

    
    #feature zusammenbauen: Anzahl Fall für jedes Bundesland in der KW mit/ohne Massnahmen

    feature_to_predict = np.array([[kalenderwoche, massnahmenJN, 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #Baden-Wüttemberg
                                   [kalenderwoche, massnahmenJN, 0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #Bayern
                                   [kalenderwoche, massnahmenJN, 0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0], #Berlin
                                   [kalenderwoche, massnahmenJN, 0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0], #Brandenburg
                                   [kalenderwoche, massnahmenJN, 0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], #Bremen
                                   [kalenderwoche, massnahmenJN, 0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0], #Hamburg
                                   [kalenderwoche, massnahmenJN, 0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0], #Hessen
                                   [kalenderwoche, massnahmenJN, 0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0], #Mecklenburg-Vorpommern
                                   [kalenderwoche, massnahmenJN, 0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0], #Niedersachsen
                                   [kalenderwoche, massnahmenJN, 0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0], #Nordrhein-Westfalen
                                   [kalenderwoche, massnahmenJN, 0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0], #Rheinland-Pfalz
                                   [kalenderwoche, massnahmenJN, 0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0], #Saarland 
                                   [kalenderwoche, massnahmenJN, 0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0], #Sachsen
                                   [kalenderwoche, massnahmenJN, 0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0], #Sachsen-Anhalt
                                   [kalenderwoche, massnahmenJN, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0], #Schleswig-Holstein
                                   [kalenderwoche, massnahmenJN, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]])#Thüringen
    
    #Lineare Regression
    labels_pred = linreg.predict(X = feature_to_predict)
    label_fuer_fehler_linReg = linreg.predict(X=features_test)
    printData(labels_pred, str_to_predict, 'Lineare Regression', kalenderwoche)
    print('Metrik der linearen Regression: ')
    print('    Score: ' + str(linreg.score(features_test, labels_test))) #Score: andere Metrik, die nicht behandelt wurde. Ähnlich zu MSE
    print('    MSE: ' + str(mean_squared_error(labels_test, label_fuer_fehler_linReg)))
    print('    MAE: ' + str(mean_absolute_error(labels_test, label_fuer_fehler_linReg)))
    #Baum
    reg_tree = tree.DecisionTreeRegressor()
    reg_tree = reg_tree.fit(features_train, labels_train)

    labels_pred_tree = reg_tree.predict(X = feature_to_predict)
    label_fuer_fehler_tree = reg_tree.predict(X=features_test)
    printData(labels_pred_tree, str_to_predict, 'Baum', kalenderwoche)
    print('Metrik Baum: ')
    print('    Score: ' + str(reg_tree.score(features_test, labels_test))) 
    print('    MSE: ' + str(mean_squared_error(labels_test, label_fuer_fehler_tree)))
    print('    MAE: ' + str(mean_absolute_error(labels_test, label_fuer_fehler_tree)))

    #neuronales Netz
    #Labelvorhersage wird größer bzw kleiner, wenn HiddenLayer größer / kleiner
    #Activation sagt geringe Werte voraus
    #plotten der Loss-Kurve funktioniert nur beim Default Optimizer
    mlp_regr = MLPRegressor(activation='relu',random_state=1, max_iter=500, solver ='lbfgs', learning_rate='invscaling').fit(features_train, labels_train)
    mlp_labels_predict = mlp_regr.predict(X = feature_to_predict)
    test_prediction = mlp_regr.predict(X = features_test)
    printData(mlp_labels_predict, str_to_predict, 'Neuronales Netz', kalenderwoche)
    print('Metrik neuronales Netz: ')
    print('Score: '+ str(mlp_regr.score(features_test, labels_test)))
    print('Loss: '+str(mlp_regr.loss_))
    kalenderwoche_to_plot = features_test[:,0]
    #Plotten der ganzen Vorraussagen und tatsächlichen Werte
    plt.scatter(kalenderwoche_to_plot, labels_test, color= 'blue')
    plt.scatter(kalenderwoche_to_plot, test_prediction, color = 'red')
    plt.ylabel(str_to_predict)
    plt.xlabel('Kalenderwoche')
    plt.title('Tatsächliche und vorausgesagte '+str_to_predict+' neuronales Netz \n von allen Bundesländern (Blau: Tatsächlich, Rot: Vorausgesagt)')
   
    plt.savefig("Result\\" + 'onehot_Bundesland_knn_linreg_linBaum_neuronales_Netz_alle_bundesländer'+ ".png")
    plt.show()
   
def predict_Data(covid_data, kalenderwoche, data_to_predict, massnahmenJN):
    print('Vorhersage von '+data_to_predict+ 'für alle Bundesländer mit der linearen Regression, dem Baum und dem neuronalen Netzwerk anhand von Onehot-coding der Bundesländer.')
    dataframe = prepareData(covid_data)
    predictData(dataframe, kalenderwoche, data_to_predict, massnahmenJN)
    
