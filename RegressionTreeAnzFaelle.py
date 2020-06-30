#Gerriet 
import numpy as np
#import DataReader as reader
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#covid_data = reader.get_covid_data()
def regressionTree(covid_data):
    covid_data = covid_data.groupby(['Meldedatum', 'Geschlecht', 'Altersgruppe']).sum() #Gruppieren der Daten + Summieren der Fälle
    
    values = np.array(covid_data.index.get_level_values(1)) #Index für Geschlecht
    values_alter = np.array(covid_data.index.get_level_values(2)) #Index für Altergruppe
    
    #integer encode
    label_encoder = LabelEncoder() #sklearn - 0 = männlich, 1 = weiblich, 2 = unbekannt
    integer_encoded = label_encoder.fit_transform(values)
    integer_encoded_alter = label_encoder.fit_transform(values_alter)
    
    #binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    integer_encoded_alter = integer_encoded_alter.reshape(len(integer_encoded_alter), 1)
    onehot_encoded_alter = onehot_encoder.fit_transform(integer_encoded_alter)
    
    #Onehot encoded in Daten laden
    covid_data.loc[:,('maennlich')] = onehot_encoded[ :, 0]
    covid_data.loc[:,('weiblich')] = onehot_encoded[ :, 1]
    covid_data.loc[:,('unbekannt')] = onehot_encoded[ :, 2]
    covid_data.loc[:,('A00-A04')] = onehot_encoded_alter[ :, 0]
    covid_data.loc[:,('A05-A14')] = onehot_encoded_alter[ :, 1]
    covid_data.loc[:,('A15-A34')] = onehot_encoded_alter[ :, 2]
    covid_data.loc[:,('A35-A59')] = onehot_encoded_alter[ :, 3]
    covid_data.loc[:,('A60-A79')] = onehot_encoded_alter[ :, 4]
    covid_data.loc[:,('A80+')] = onehot_encoded_alter[ :, 5]
    covid_data.loc[:,('A_unbekannt')] = onehot_encoded_alter[ :, 6]
    
    #Löschen unnötiger Daten
    covid_data = covid_data.drop(['AnzahlTodesfall'], 1)
    covid_data = covid_data.drop(['AnzahlGenesen'], 1)
    
    
    features = np.array(covid_data.drop(['AnzahlFall'], 1)) #drop weil Anzahl unser Label
    features = preprocessing.scale(features)
    labels = np.array(covid_data.filter(items=['AnzahlFall']))
    
    
    reg_tree = tree.DecisionTreeRegressor()
    fig, ax = plt.subplots(figsize=(27,15)) #Größe in Inch des Bildes
    tree.plot_tree(reg_tree.fit(features, labels), max_depth=10, fontsize=10) #max_depth = Tiefe des Baums
    plt.title("Regressiver Baum über die Anzahl der Fälle bzgl. Alter und Geschlecht")  
    fig.savefig("Result\\" + 'Linearer_Baum_Fälle_Geschlecht_Altersgruppe' + ".png")
    plt.show()
    
    #Männlich und alle Altersgruppen
    featureToPredict = np.array([[1,0,0, 1,0,0,0,0,0,0],
                                 [1,0,0, 0,1,0,0,0,0,0],
                                 [1,0,0, 0,0,1,0,0,0,0],
                                 [1,0,0, 0,0,0,1,0,0,0],
                                 [1,0,0, 0,0,0,0,1,0,0],
                                 [1,0,0, 0,0,0,0,0,1,0]])
    labels_pred = reg_tree.predict(featureToPredict)
    print('Vorhergesagte Anzahl Fall pro Tag : \n'+
          'für männlich: \n'+
          'für Altersgruppe 00-04 ' + str(labels_pred[0])+ '\n' +
          'für Altersgruppe 05-14 ' + str(labels_pred[1])+'\n' +
          'für Altersgruppe 15-34 ' + str(labels_pred[2])+'\n' +
          'für Altersgruppe 35-59 ' + str(labels_pred[3])+'\n' +
          'für Altersgruppe 60-79 ' + str(labels_pred[4])+'\n' +
          'für Altersgruppe 80+ ' + str(labels_pred[5]))
    
    
    #Weiblich und alle Altersgruppen
    featureToPredict = np.array([[0,1,0, 1,0,0,0,0,0,0],
                                 [0,1,0, 0,1,0,0,0,0,0],
                                 [0,1,0, 0,0,1,0,0,0,0],
                                 [0,1,0, 0,0,0,1,0,0,0],
                                 [0,1,0, 0,0,0,0,1,0,0],
                                 [0,1,0, 0,0,0,0,0,1,0]])
    labels_pred = reg_tree.predict(featureToPredict)
    print('Vorhergesagte Anzahl Fall pro Tag : \n'+
          'für weiblich: \n'+
          'für Altersgruppe 00-04 ' + str(labels_pred[0])+ '\n' +
          'für Altersgruppe 05-14 ' + str(labels_pred[1])+'\n' +
          'für Altersgruppe 15-34 ' + str(labels_pred[2])+'\n' +
          'für Altersgruppe 35-59 ' + str(labels_pred[3])+'\n' +
          'für Altersgruppe 60-79 ' + str(labels_pred[4])+'\n' +
          'für Altersgruppe 80+ ' + str(labels_pred[5]))