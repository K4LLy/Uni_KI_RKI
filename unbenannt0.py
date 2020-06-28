import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np

print('Reading Wetterdaten - mittlere Lufttemperatur Niedersachsen...')
data_lufttemp = pd.read_csv('Data\\Wetterdaten\\mittlere_Lufttemperatur\\data\\data_TMK_MN004.csv')
data_stationen = pd.read_csv('Data\\Wetterdaten\\mittlere_Lufttemperatur\\data\\sdo_TMK_MN004.csv')
print('Wetterdaten - mittlere Lufttemperatur Niedersachsen read.')
 
data_nds_wetter = data_lufttemp.merge(data_stationen, on='SDO_ID') 
print(len(data_nds_wetter["SDO_Name"].unique()))

ListeCovid.landkreis.lat
ListeCovid.landkreis.lon

ListeWetter.Station.lat
ListeWetter.Station.lon

ListeCovid erweitern um Wetterinformationsspalten
foreach eintrag in ListeCovid
    geo_inf_lk = geopy.location(eintrag['Landkreis'])
    Wetterinfos = coole_funktion(geo_inf_lk, ListeWetter)
    ListerCovid[Wetterspalten] = Wetterinfos


df_list = []

for index, row in data_nds_wetter.iterrows():
    datum = row['Zeitstempel']
    zeitstempel = dt.datetime.strptime(str(datum), '%Y%m%d').strftime('%d/%m/%Y')
    df_list.append([row['SDO_ID'], row['SDO_Name'], zeitstempel, row['Wert']])
    
data_nds = pd.DataFrame(df_list, columns=['SDO_ID', 'Messstation', 'Datum', 'Temperatur'])

messstation = data_nds['Messstation']=='Diepholz'
data_messstation = data_nds[messstation]

plt.plot(data_messstation.Datum, data_messstation.Temperatur)
plt.title('Temperaturverlauf in Oldenburg')
plt.xlabel('Datum')
plt.ylabel('Temperatur in °C')
plt.show()