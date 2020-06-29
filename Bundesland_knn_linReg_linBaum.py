

#import
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as tree_plt
import matplotlib.pyplot as lin_plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
from sklearn.neural_network import MLPRegressor
import Util as u


def prepareData(covid_data, bundesland_to_filter):
    print("Prepare Data...")
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
    maskenpflicht = []
    kontaktbeschraenkung = []
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
        if bundesland_to_filter == 'Baden-Württemberg':
            if kalenderwoche_nr >= 18:
                maskenpflicht.append(1)
            else:
                maskenpflicht.append(0)
            if kalenderwoche_nr >= 17 and kalenderwoche_nr < 26:
                kontaktbeschraenkung.append(1)
            else:
                kontaktbeschraenkung.append(0)
        elif bundesland_to_filter == 'Bayern':
            if kalenderwoche_nr >= 18:
                maskenpflicht.append(1)
            else:
                maskenpflicht.append(0)
            if kalenderwoche_nr >= 17 and kalenderwoche_nr < 26:
                kontaktbeschraenkung.append(1)
            else:
                kontaktbeschraenkung.append(0)
        elif bundesland_to_filter == 'Berlin':
            if kalenderwoche_nr >= 18:
                maskenpflicht.append(1)
            else:
                maskenpflicht.append(0)
            if kalenderwoche_nr >= 17:
                kontaktbeschraenkung.append(1)
            else:
                kontaktbeschraenkung.append(0)
        elif bundesland_to_filter == 'Brandenburg':
            if kalenderwoche_nr >= 18:
                maskenpflicht.append(1)
            else:
                maskenpflicht.append(0)
            if kalenderwoche_nr >= 17 and kalenderwoche_nr < 25:
                kontaktbeschraenkung.append(1)
            else:
                kontaktbeschraenkung.append(0)
        elif bundesland_to_filter == 'Bremen':
            if kalenderwoche_nr >= 18:
                maskenpflicht.append(1)
            else:
                maskenpflicht.append(0)
            if kalenderwoche_nr >= 17 and kalenderwoche_nr < 26:
                kontaktbeschraenkung.append(1)
            else:
                kontaktbeschraenkung.append(0)
        elif bundesland_to_filter == 'Hamburg':
            if kalenderwoche_nr >= 18:
                maskenpflicht.append(1)
            else:
                maskenpflicht.append(0)
            if kalenderwoche_nr >= 17 and kalenderwoche_nr < 26:
                kontaktbeschraenkung.append(1)
            else:
                kontaktbeschraenkung.append(0)
        elif bundesland_to_filter == 'Hessen':
            if kalenderwoche_nr >= 18:
                maskenpflicht.append(1)
            else:
                maskenpflicht.append(0)
            if kalenderwoche_nr >= 17 and kalenderwoche_nr < 24:
                kontaktbeschraenkung.append(1)
            else:
                kontaktbeschraenkung.append(0)
        elif bundesland_to_filter == 'Mecklenburg-Vorpommern':
            if kalenderwoche_nr >= 18:
                maskenpflicht.append(1)
            else:
                maskenpflicht.append(0)
            if kalenderwoche_nr >= 17 and kalenderwoche_nr < 26:
                kontaktbeschraenkung.append(1)
            else:
                kontaktbeschraenkung.append(0)
        elif bundesland_to_filter == 'Niedersachsen':
            if kalenderwoche_nr >= 18:
                maskenpflicht.append(1)
            else:
                maskenpflicht.append(0)
            if kalenderwoche_nr >= 17 and kalenderwoche_nr < 26:
                kontaktbeschraenkung.append(1)
            else:
                kontaktbeschraenkung.append(0)
        elif bundesland_to_filter == 'Nordrhein-Westfalen':
            if kalenderwoche_nr >= 18:
                maskenpflicht.append(1)
            else:
                maskenpflicht.append(0)
            if kalenderwoche_nr >= 17 and kalenderwoche_nr < 26:
                kontaktbeschraenkung.append(1)
            else:
                kontaktbeschraenkung.append(0)
        elif bundesland_to_filter == 'Rheinland-Pfalz':
            if kalenderwoche_nr >= 18:
                maskenpflicht.append(1)
            else:
                maskenpflicht.append(0)
            if kalenderwoche_nr >= 17 and kalenderwoche_nr < 26:
                kontaktbeschraenkung.append(1)
            else:
                kontaktbeschraenkung.append(0)
        elif bundesland_to_filter == 'Saarland':
            if kalenderwoche_nr >= 18:
                maskenpflicht.append(1)
            else:
                maskenpflicht.append(0)
            if kalenderwoche_nr >= 17 and kalenderwoche_nr < 26:
                kontaktbeschraenkung.append(1)
            else:
                kontaktbeschraenkung.append(0)
        elif bundesland_to_filter == 'Sachsen':
            if kalenderwoche_nr >= 17:
                maskenpflicht.append(1)
            else:
                maskenpflicht.append(0)
            if kalenderwoche_nr >= 17 and kalenderwoche_nr < 24:
                kontaktbeschraenkung.append(1)
            else:
                kontaktbeschraenkung.append(0)
        elif bundesland_to_filter == 'Sachsen-Anhalt':
            if kalenderwoche_nr >= 17:
                maskenpflicht.append(1)
            else:
                maskenpflicht.append(0)
            if kalenderwoche_nr >= 17 and kalenderwoche_nr < 23:
                kontaktbeschraenkung.append(1)
            else:
                kontaktbeschraenkung.append(0)
        elif bundesland_to_filter == 'Schleswig-Holstein':
            if kalenderwoche_nr >= 18:
                maskenpflicht.append(1)
            else:
                maskenpflicht.append(0)
            if kalenderwoche_nr >= 17 and kalenderwoche_nr < 26:
                kontaktbeschraenkung.append(1)
            else:
                kontaktbeschraenkung.append(0)
        elif bundesland_to_filter == 'Thüringen':
            if kalenderwoche_nr >= 17:
                maskenpflicht.append(1)
            else:
                maskenpflicht.append(0)
            if kalenderwoche_nr >= 17 and kalenderwoche_nr < 24:
                kontaktbeschraenkung.append(1)
            else:
                kontaktbeschraenkung.append(0)
        else:
            maskenpflicht.append(0)
            kontaktbeschraenkung.append(0)
            
       
        
    
        
    covid_data.loc[:,('GroßveranstaltungJN')] = massnahmen
    covid_data.loc[:,('MaskenpflichtJN')] = maskenpflicht
    covid_data.loc[:,('KontaktbeschraenkungJN')] = kontaktbeschraenkung #Treffen von bis zu 10 Personen gilt hier als keine Kontaktbeschränkung
    covid_data.loc[:,('Kalenderwoche')] = kalenderwochen_nr
    covid_data.loc[:,('R_Null_Faktor')] = r_null_faktor
    covid_data = covid_data.loc[covid_data[bundesland_to_filter] == 1] #Filter nach gefragtem Bundesland
    print("Data prepared")
    return covid_data

def printData(labels_pred, str_to_predict, regression, kalenderwoche, bundesland):
    print(regression +' '+str_to_predict + ' in KW '+str(kalenderwoche) +' :' + '\n' +
          str_to_predict + ' in '  +bundesland  + ': '   + str(labels_pred[0])
          )
    
def predictData(covid_data, kalenderwoche, column_to_predict, massnahmenJN, bundesland,maskeJN, kontaktJN):
   
    #label: anzahl fall 
    dataframe_der_features = covid_data.filter(items= ['Kalenderwoche', 'GroßveranstaltungJN', 'MaskenpflichtJN', 'KontaktbeschraenkungJN' ])
    features = np.array(dataframe_der_features)
 
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

    
    #feature zusammenbauen:  KW mit/ohne Massnahmen

    feature_to_predict = np.array([[kalenderwoche, massnahmenJN, maskeJN, kontaktJN]])
    
    #Lineare Regression
    labels_pred = linreg.predict(X = feature_to_predict)
    label_fuer_fehler_linReg = linreg.predict(X=features_test)
    test_prediction = linreg.predict(X=features)
    printData(labels_pred, str_to_predict, 'Lineare Regression', kalenderwoche, bundesland)
    print('Metrik der linearen Regression: ')
    print('    Score: ' + str(linreg.score(features_test, labels_test))) #Score: andere Metrik, die nicht behandelt wurde. Ähnlich zu MSE
    print('    MSE: ' + str(mean_squared_error(labels_test, label_fuer_fehler_linReg)))
    print('    MAE: ' + str(mean_absolute_error(labels_test, label_fuer_fehler_linReg)))
    
    kalenderwoche_to_plot = features[:,0]
    lin_plt.scatter(kalenderwoche_to_plot, labels, color= 'blue')
    lin_plt.scatter(kalenderwoche_to_plot, test_prediction, color = 'red')
    lin_plt.ylabel(str_to_predict)
    lin_plt.xlabel('Kalenderwoche')
    lin_plt.title('Tatsächliche und vorausgesagte '+str_to_predict+' lineare Regression \n mit einem Bundesland für ' +bundesland + '(Blau: Tatsächlich, Rot: Vorausgesagt)')
    
    lin_plt.savefig("Result\\" + 'Bundesland_knn_linReg_linBaum_lineare_Regression_' +bundesland+ '_'+str_to_predict+".png")
    lin_plt.show()
    #Baum
    reg_tree = tree.DecisionTreeRegressor()
    reg_tree = reg_tree.fit(features_train, labels_train)

    labels_pred_tree = reg_tree.predict(X = feature_to_predict)
    label_fuer_fehler_tree = reg_tree.predict(X=features_test)
    test_prediction = reg_tree.predict(X=features_test)
    printData(labels_pred_tree, str_to_predict, 'Baum', kalenderwoche, bundesland)
    print('Metrik Baum: ')
    print('    Score: ' + str(reg_tree.score(features_test, labels_test))) 
    print('    MSE: ' + str(mean_squared_error(labels_test, label_fuer_fehler_tree)))
    print('    MAE: ' + str(mean_absolute_error(labels_test, label_fuer_fehler_tree)))
    kalenderwoche_to_plot = features_test[:,0]
    tree_plt.scatter(kalenderwoche_to_plot, labels_test, color= 'blue')
    tree_plt.scatter(kalenderwoche_to_plot, test_prediction, color = 'red')
    tree_plt.ylabel(str_to_predict)
    tree_plt.xlabel('Kalenderwoche')
    tree_plt.title('Tatsächliche und vorausgesagte '+str_to_predict+' Baum \n mit einem Bundesland für ' +bundesland + '(Blau: Tatsächlich, Rot: Vorausgesagt)')
    
    tree_plt.savefig("Result\\" + 'Bundesland_knn_linReg_linBaum_Baum_' +bundesland+ '_'+str_to_predict+".png")
    tree_plt.show()
    #neuronales Netz
    
    my_hiddenlayer_size = (80,)
    mlp_regr = MLPRegressor(hidden_layer_sizes= my_hiddenlayer_size, activation='relu',random_state=1, max_iter=500, solver ='lbfgs', learning_rate='constant').fit(features_train, labels_train)
    
    mlp_labels_predict = mlp_regr.predict(X = feature_to_predict)
    test_prediction = mlp_regr.predict(X= features)
    printData(mlp_labels_predict, str_to_predict, 'Neuronales Netz', kalenderwoche, bundesland)
    print('Metrik neuronales Netz: ')
    print('    Score: '+ str(mlp_regr.score(features_test, labels_test)))
    print('    Loss: '+str(mlp_regr.loss_))
    #zum Plotten der Loss-Kurve:
    #pd.DataFrame(mlp_regr.loss_curve_).plot()
    kalenderwoche_to_plot = features[:,0]
    plt.scatter(kalenderwoche_to_plot, labels, color= 'blue')
    plt.scatter(kalenderwoche_to_plot, test_prediction, color = 'red')
    plt.ylabel(str_to_predict)
    plt.xlabel('Kalenderwoche')
    plt.title('Tatsächliche und vorausgesagte '+str_to_predict+' neuronales Netz \n mit einem Bundesland für ' +bundesland + ' (Blau: Tatsächlich, Rot: Vorausgesagt)')
    
    plt.savefig("Result\\" + 'Bundesland_knn_linReg_linBaum_neuronales_Netz_' +bundesland+ '_'+str_to_predict+".png")
    plt.show()
    u.save_model(mlp_regr, 'Bundesland_knn_linReg_linBaum_hiddenlayersize_'+str(my_hiddenlayer_size)+'_'+str_to_predict)
   
def predict_Data(covid_data, kalenderwoche, data_to_predict, massnahmenJN, bundesland,maskeJN, kontaktJN):
    print('Vorhersage von '+data_to_predict+ 'für das Bundesland '+ bundesland + 'mit der linearen Regression, dem Baum und dem neuronalen Netzwerk.')
    dataframe = prepareData(covid_data, bundesland)
    predictData(dataframe, kalenderwoche, data_to_predict, massnahmenJN, bundesland,maskeJN, kontaktJN)

    


