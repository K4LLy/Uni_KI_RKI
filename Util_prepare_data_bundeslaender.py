#Lara Ahrens
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

def onehot_encode_data(covid_data):
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

    return covid_data
def prepare_Data_for_one_bundesland(covid_data, bundesland_to_filter):
    

    #ab kalenderwoche 12 werden Maßnahmen getroffen: keine Großveranstaltungen mehr
    r_null_faktor = []
    massnahmen = []
    maskenpflicht = []
    kontaktbeschraenkung = []
    kalenderwochen_nr = []
    for indexes, row in covid_data.iterrows():
        kalenderwoche = row['KW']
        anzahlFall = row ['AnzahlFall']
        anzahlGenesen = row['AnzahlGenesen']
        #r-null-Faktor in den Kalenderwochen
        if anzahlGenesen != 0:
            r_null = anzahlFall / anzahlGenesen
            r_null_faktor.append(r_null)
        else:
            r_null_faktor.append(anzahlFall)
        kalenderwoche_nr = int(kalenderwoche)
        kalenderwochen_nr.append(kalenderwoche_nr) # Kalenderwoche ist nur als String verfügbar --> Integer für Feature nötig
        if kalenderwoche_nr >= 12:
            massnahmen.append(1)
        else:
            massnahmen.append(0)
            #Maskenpflicht und Kontaktbeschränkungen in den einzelnen Bundesländern
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
            
       
        
    
    #Daten ins Dataframe hinzufügen   
    covid_data.loc[:,('GroßveranstaltungJN')] = massnahmen
    covid_data.loc[:,('MaskenpflichtJN')] = maskenpflicht
    covid_data.loc[:,('KontaktbeschraenkungJN')] = kontaktbeschraenkung #Treffen von bis zu 10 Personen gilt hier als keine Kontaktbeschränkung
    covid_data.loc[:,('Kalenderwoche')] = kalenderwochen_nr
    covid_data.loc[:,('R_Null_Faktor')] = r_null_faktor
    covid_data = covid_data.loc[covid_data[bundesland_to_filter] == 1] #Filter nach gefragtem Bundesland
    print("Data prepared")
    return covid_data

def prepare_data_for_every_bundesland(covid_data):
     #Bezug der Maßnahmen auf ganz Deutschland - Durchschnittswerte
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
        if kalenderwoche_nr >= 18:
            maskenpflicht.append(1)
        else:
            maskenpflicht.append(0)
        if kalenderwoche_nr >= 17 and kalenderwoche_nr < 26:
            kontaktbeschraenkung.append(1)
        else:
            kontaktbeschraenkung.append(0)
   
    covid_data.loc[:,('GroßveranstaltungJN')] = massnahmen
    covid_data.loc[:,('MaskenpflichtJN')] = maskenpflicht
    covid_data.loc[:,('KontaktbeschraenkungJN')] = kontaktbeschraenkung #Treffen von bis zu 10 Personen gilt hier als keine Kontaktbeschränkung
    covid_data.loc[:,('Kalenderwoche')] = kalenderwochen_nr
    covid_data.loc[:,('Kalenderwoche_for_filter')] = kalenderwochen_nr
    covid_data.loc[:,('R_Null_Faktor')] = r_null_faktor
    
    print("Data prepared")
    return covid_data