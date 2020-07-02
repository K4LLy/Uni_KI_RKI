#Lara Ahrens

#import
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt8
import matplotlib.pyplot as tree_plt
import matplotlib.pyplot as lin_plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import tree
from sklearn.neural_network import MLPRegressor
import Util_prepare_data_bundeslaender as preparing
import Util_print_data_bundeslaender as printing


    
def predict_Data_for_onehot_encoded_bundesland(covid_data, kalenderwoche, column_to_predict, massnahmenJN, maskejn, kontaktjn):
    str_to_predict = ''
    if column_to_predict == 'AnzahlFall':
       str_to_predict = 'Anzahl der Fälle' 
    elif column_to_predict =='AnzahlGenesen':
        str_to_predict = 'Gesunde Fälle'
    elif column_to_predict =='AnzahlTodesfall':
        str_to_predict = 'Tote Fälle'
    elif column_to_predict == 'R_Null_Faktor':
        str_to_predict = 'R0-Faktor'
    print('Vorhersage von '+str_to_predict+ 'für alle Bundesländer mit der linearen Regression, dem Baum und dem neuronalen Netzwerk anhand von Onehotencoding der Bundesländer.')
    covid_data = preparing.onehot_encode_data(covid_data)
    covid_data = preparing.prepare_data_for_every_bundesland(covid_data)

    #features: Kalenderwoche, Bundesland (dargestellt als 0 und 1), massnahmen
    #label: colum_to_predict (AnzahlFall/Todesfall/genesen)
    dataframe_der_features = covid_data.filter(items= ['Kalenderwoche', 'GroßveranstaltungJN', 'MaskenpflichtJN','KontaktbeschraenkungJN', 'Baden-Württemberg', 'Bayern', 'Berlin', 'Brandenburg','Bremen', 'Hamburg', 'Hessen', 'Mecklenburg-Vorpommern','Niedersachsen', 'Nordrhein-Westfalen', 'Rheinland-Pfalz', 'Saarland', 'Sachsen', 'Sachsen-Anhalt', 'Schleswig-Holstein', 'Thüringen' ])
    features = np.array(dataframe_der_features)

    dataframe_der_labels = covid_data.filter(items = [column_to_predict])
    
    labels = np.array(dataframe_der_labels)
    #Test und Trainingssatz

    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)
    #lineare Regression zusammenbauen und trainieren
    linreg = LinearRegression()
    linreg.fit(features_train, labels_train)

    
    #feature zusammenbauen: Anzahl Fall für jedes Bundesland in der KW mit/ohne Massnahmen

    feature_to_predict = np.array([[kalenderwoche, massnahmenJN, maskejn, kontaktjn, 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #Baden-Wüttemberg
                                   [kalenderwoche, massnahmenJN, maskejn, kontaktjn, 0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #Bayern
                                   [kalenderwoche, massnahmenJN, maskejn, kontaktjn, 0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0], #Berlin
                                   [kalenderwoche, massnahmenJN, maskejn, kontaktjn, 0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0], #Brandenburg
                                   [kalenderwoche, massnahmenJN, maskejn, kontaktjn, 0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], #Bremen
                                   [kalenderwoche, massnahmenJN, maskejn, kontaktjn, 0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0], #Hamburg
                                   [kalenderwoche, massnahmenJN, maskejn, kontaktjn, 0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0], #Hessen
                                   [kalenderwoche, massnahmenJN, maskejn, kontaktjn, 0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0], #Mecklenburg-Vorpommern
                                   [kalenderwoche, massnahmenJN, maskejn, kontaktjn, 0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0], #Niedersachsen
                                   [kalenderwoche, massnahmenJN, maskejn, kontaktjn, 0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0], #Nordrhein-Westfalen
                                   [kalenderwoche, massnahmenJN, maskejn, kontaktjn, 0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0], #Rheinland-Pfalz
                                   [kalenderwoche, massnahmenJN, maskejn, kontaktjn, 0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0], #Saarland 
                                   [kalenderwoche, massnahmenJN, maskejn, kontaktjn, 0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0], #Sachsen
                                   [kalenderwoche, massnahmenJN, maskejn, kontaktjn, 0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0], #Sachsen-Anhalt
                                   [kalenderwoche, massnahmenJN, maskejn, kontaktjn, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0], #Schleswig-Holstein
                                   [kalenderwoche, massnahmenJN, maskejn, kontaktjn, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]])#Thüringen
    
    #Lineare Regression
    labels_pred = linreg.predict(X = feature_to_predict)
    label_fuer_fehler_linReg = linreg.predict(X=features_test)
    printing.print_Data_every_bundesland_onehot_encoded(labels_pred, str_to_predict, 'Lineare Regression', kalenderwoche)
    print('Metrik der linearen Regression: ')
    print('    Score: ' + str(linreg.score(features_test, labels_test))) #Score: andere Metrik, die nicht behandelt wurde. Ähnlich zu MSE
    print('    MSE: ' + str(mean_squared_error(labels_test, label_fuer_fehler_linReg)))
    print('    MAE: ' + str(mean_absolute_error(labels_test, label_fuer_fehler_linReg)))
    #Baum
    reg_tree = tree.DecisionTreeRegressor()
    reg_tree = reg_tree.fit(features_train, labels_train)

    labels_pred_tree = reg_tree.predict(X = feature_to_predict)
    label_fuer_fehler_tree = reg_tree.predict(X=features_test)
    printing.print_Data_every_bundesland_onehot_encoded(labels_pred_tree, str_to_predict, 'Baum', kalenderwoche)
    print('Metrik Baum: ')
    print('    Score: ' + str(reg_tree.score(features_test, labels_test))) 
    print('    MSE: ' + str(mean_squared_error(labels_test, label_fuer_fehler_tree)))
    print('    MAE: ' + str(mean_absolute_error(labels_test, label_fuer_fehler_tree)))

    #neuronales Netz
    #Vorhersage wird größer bzw kleiner, wenn HiddenLayer größer / kleiner
    #Activation adan verursacht einen linearen Verlauf
    #plotten der Loss-Kurve funktioniert nur beim Default Optimizer adam
   
    mlp_regr = MLPRegressor(activation='relu',random_state=1, max_iter=500, solver ='lbfgs', learning_rate='invscaling').fit(features_train, labels_train)
    mlp_labels_predict = mlp_regr.predict(X = feature_to_predict)
    test_prediction = mlp_regr.predict(X = features_test)
    printing.print_Data_every_bundesland_onehot_encoded(mlp_labels_predict, str_to_predict, 'Neuronales Netz', kalenderwoche)
    print('Metrik neuronales Netz: ')
    print('Score: '+ str(mlp_regr.score(features_test, labels_test)))
    print('Loss: '+str(mlp_regr.loss_))
    kalenderwoche_to_plot = features_test[:,0]
    #Plotten der ganzen Vorraussagen und tatsächlichen Werte
    plt.scatter(kalenderwoche_to_plot, labels_test, color= 'blue')
    plt.scatter(kalenderwoche_to_plot, test_prediction, color = 'red')
    plt.ylabel(str_to_predict)
    plt.xlabel('Kalenderwoche')
    plt.title('Neuronales Netz (alle Bundesländer (one-hot)) \n (Blau: tats., Rot: voraus.)')
   
    plt.savefig("Result\\" + 'onehot_Bundesland_knn_linreg_linBaum_neuronales_Netz_alle_bundesländer'+ ".png")
    plt.show()
    


def predict_Data_for_one_bundesland(covid_data, kalenderwoche, column_to_predict, massnahmenJN, bundesland,maskeJN, kontaktJN):
    str_to_predict = ''
    if column_to_predict == 'AnzahlFall':
       str_to_predict = 'Anzahl der Fälle' 
    elif column_to_predict =='AnzahlGenesen':
        str_to_predict = 'Gesunde Fälle'
    elif column_to_predict =='AnzahlTodesfall':
        str_to_predict = 'Tote Fälle'
    elif column_to_predict == 'R_Null_Faktor':
        str_to_predict = 'R0-Faktor'
    print('Vorhersage von '+str_to_predict+ ' für das Bundesland '+ bundesland + ' mit der linearen Regression, dem Baum und dem neuronalen Netzwerk.')
    covid_data = preparing.onehot_encode_data(covid_data)
    covid_data = preparing.prepare_Data_for_one_bundesland(covid_data, bundesland)
    #Features
    dataframe_der_features = covid_data.filter(items= ['Kalenderwoche', 'GroßveranstaltungJN', 'MaskenpflichtJN', 'KontaktbeschraenkungJN' ])
    features = np.array(dataframe_der_features)
    #Label: AnzahlFall/AnzahlTodesfall/AnzahlGenesen/R-0_faktor im übergebenen Bundesland
    dataframe_der_labels = covid_data.filter(items = [column_to_predict])
    
    labels = np.array(dataframe_der_labels)
    #Test und Trainingssatz

    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)
    #lineare Regression zusammenbauen und trainieren
    linreg = LinearRegression()
    linreg.fit(features_train, labels_train)

    
    #feature zusammenbauen, das von dem Model predicted werden soll:  KW mit/ohne Massnahmen

    feature_to_predict = np.array([[kalenderwoche, massnahmenJN, maskeJN, kontaktJN]])
    
    #Lineare Regression
    labels_pred = linreg.predict(X = feature_to_predict)
    label_fuer_fehler_linReg = linreg.predict(X=features_test)
    test_prediction = linreg.predict(X=features)
    printing.print_Data_for_one_bundesland(labels_pred, str_to_predict, 'Lineare Regression', kalenderwoche, bundesland)
    print('Metrik der linearen Regression: ')
    print('    Score: ' + str(linreg.score(features_test, labels_test))) #Score: andere Metrik, die nicht behandelt wurde. Ähnlich zu MSE
    print('    MSE: ' + str(mean_squared_error(labels_test, label_fuer_fehler_linReg)))
    print('    MAE: ' + str(mean_absolute_error(labels_test, label_fuer_fehler_linReg)))
    
    #Tatsächliche Werte und von der linearen Regression vorausgesagte Werte für das Bundesland von allen vorhandenen Werten
    kalenderwoche_to_plot = features[:,0]
    lin_plt.scatter(kalenderwoche_to_plot, labels, color= 'blue')
    lin_plt.scatter(kalenderwoche_to_plot, test_prediction, color = 'red')
    lin_plt.ylabel(str_to_predict)
    lin_plt.xlabel('Kalenderwoche')
    lin_plt.title('Lineare Regression mit einem Bundesland für ' +bundesland + '\n(Blau: Tatsächlich, Rot: Vorausgesagt)')
    
    lin_plt.savefig("Result\\" + 'Bundesland_knn_linReg_linBaum_lineare_Regression_' +bundesland+ '_'+str_to_predict+".png")
    lin_plt.show()
    
    #Baum
    reg_tree = tree.DecisionTreeRegressor()
    reg_tree = reg_tree.fit(features_train, labels_train)

    labels_pred_tree = reg_tree.predict(X = feature_to_predict)
    label_fuer_fehler_tree = reg_tree.predict(X=features_test)
    test_prediction = reg_tree.predict(X=features_test)
    printing.print_Data_for_one_bundesland(labels_pred_tree, str_to_predict, 'Baum', kalenderwoche, bundesland)
    print('Metrik Baum: ')
    print('    Score: ' + str(reg_tree.score(features_test, labels_test))) 
    print('    MSE: ' + str(mean_squared_error(labels_test, label_fuer_fehler_tree)))
    print('    MAE: ' + str(mean_absolute_error(labels_test, label_fuer_fehler_tree)))
    
    
    #Tatsächliche Werte und von dem Baum vorausgesagte Werte für das Bundesland von allen Testwerten. Die Trainingswerte würden auf jeden Fall zum 100%tigen Ergebnis führen!

    kalenderwoche_to_plot = features_test[:,0]
    tree_plt.scatter(kalenderwoche_to_plot, labels_test, color= 'blue')
    tree_plt.scatter(kalenderwoche_to_plot, test_prediction, color = 'red')
    tree_plt.ylabel(str_to_predict)
    tree_plt.xlabel('Kalenderwoche')
    tree_plt.title('Baum mit einem Bundesland für ' +bundesland + '\n(Blau: Tatsächlich, Rot: Vorausgesagt)')
    
    tree_plt.savefig("Result\\" + 'Bundesland_knn_linReg_linBaum_Baum_' +bundesland+ '_'+str_to_predict+".png")
    tree_plt.show()
    
    #neuronales Netz
    #Optimizer Adam sagt einen linearen Verlauf der Kurve voraus
    my_hiddenlayer_size = (80,)
    mlp_regr = MLPRegressor(hidden_layer_sizes= my_hiddenlayer_size, activation='relu',random_state=1, max_iter=500, solver ='lbfgs', learning_rate='constant').fit(features_train, labels_train)
    
    mlp_labels_predict = mlp_regr.predict(X = feature_to_predict)
    test_prediction = mlp_regr.predict(X= features)
    printing.print_Data_for_one_bundesland(mlp_labels_predict, str_to_predict, 'Neuronales Netz', kalenderwoche, bundesland)
    print('Metrik neuronales Netz: ')
    print('    Score: '+ str(mlp_regr.score(features_test, labels_test)))
    print('    Loss: '+str(mlp_regr.loss_))
   
    kalenderwoche_to_plot = features[:,0]
    plt.scatter(kalenderwoche_to_plot, labels, color= 'blue')
    plt.scatter(kalenderwoche_to_plot, test_prediction, color = 'red')
    plt.ylabel(str_to_predict)
    plt.xlabel('Kalenderwoche')
    plt.title('Neuronales Netz mit einem Bundesland für ' +bundesland + ' \n (Blau: Tatsächlich, Rot: Vorausgesagt)')
    
    plt.savefig("Result\\" + 'Bundesland_knn_linReg_linBaum_neuronales_Netz_' +bundesland+ '_'+str_to_predict+".png")
    plt.show()
    # zum Speichern des Models: u.save_model(mlp_regr, 'Bundesland_knn_linReg_linBaum_hiddenlayersize_'+str(my_hiddenlayer_size)+'_'+str_to_predict)
      

def predict_data_with_knn_multi_label(covid_data, column_to_predict, kalenderwoche, grossveranstaltung,maskenpflicht,kontaktbeschraenkung):
    str_to_predict = ''
    if column_to_predict == 'AnzahlFall':
       str_to_predict = 'Anzahl der Fälle' 
    elif column_to_predict =='AnzahlGenesen':
        str_to_predict = 'Gesunde Fälle'
    elif column_to_predict =='AnzahlTodesfall':
        str_to_predict = 'Tote Fälle'
    print("Anhand eines Features wird ein Label mit den Fällen für jedes Bundesland vorausgesagt")
    covid_data = preparing.onehot_encode_data(covid_data)
    covid_data = preparing.prepare_data_for_every_bundesland(covid_data)
    covid_data = covid_data.loc[covid_data['Kalenderwoche'] >= 10  ] #nur für alle Bundesländer!
    covid_data = covid_data.loc[covid_data['Kalenderwoche'] <26  ] 
    
    #features sind für jedes Bundesland gleich, dürfen aber nur die Größe der Label haben
    features_dataframe = covid_data.loc[covid_data['Bundesland'] == 'Bayern']
    features_dataframe = features_dataframe.filter(items = ['Kalenderwoche', 'GroßveranstaltungJN', 'MaskenpflichtJN', 'KontaktbeschraenkungJN'])

    
    labels_dataframe = covid_data.filter(items = [column_to_predict])
    labels_array1 = np.array(labels_dataframe)
    
    #Das Label besteht aus einem Vektor, das für jedes Bundesland eine Ausgabe hat
    features = np.array(features_dataframe)
    labels = np.reshape(labels_array1, (-1,16))
    
    #Test- und Trainingsdaten
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)
    
    #Neuronales Netz
    #Optimizer Adam sagt einen linearen Verlauf der Kurve voraus
    my_hiddenlayer_size = (80,10)
    mlp_regr = MLPRegressor(hidden_layer_sizes= my_hiddenlayer_size, activation='relu',random_state=1, max_iter=500, solver ='lbfgs', learning_rate='constant').fit(features_train, labels_train)
    
    
    #Feature zusammebauen für die Vorraussage 
    feature_to_predict = np.array([[kalenderwoche,grossveranstaltung,maskenpflicht,kontaktbeschraenkung]])
    #Vorhersage
    labels_pred = mlp_regr.predict(X= feature_to_predict)
    printing.print_prediction_multi_label(column_to_predict, kalenderwoche, labels_pred, str_to_predict)
    #je nachdem, wie der Test- und Trainingssatz gesplittet wird, ist der Score/Loss niedriger/höher
    print('Metrik neuronales Netz: ')
    print('    Score: '+ str(mlp_regr.score(features_test, labels_test)))
    print('    Loss: '+str(mlp_regr.loss_))
    
    #für die Abbildung
    test_prediction = mlp_regr.predict(X= features)
    kalenderwoche_to_plot = features[:,0]
    plt8.scatter(kalenderwoche_to_plot, labels[:,8], color= 'blue')  
    plt8.scatter(kalenderwoche_to_plot, test_prediction[:,8], color = 'red')
    plt8.ylabel(str_to_predict)
    plt8.xlabel('Kalenderwoche')
    plt8.title('Neuronales Netz (mit Output von allen Fällen) \n für Niedersachsen (Blau: Tatsächlich, Rot: Vorausgesagt)')
      
    #Zum Speichern des Models: u.save_model(mlp_regr, 'knn_bundeslaender_hiddenlayersize_'+str(my_hiddenlayer_size))
    plt8.savefig("Result\\" + 'knn_bundeslaender_neuronales_netz_niedersachsen_'+str_to_predict + ".png")
    plt8.show()
    


