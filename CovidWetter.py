import DataReader as reader
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
from geopy import distance, Nominatim
from sklearn.neighbors import BallTree, DistanceMetric
geolocator = Nominatim(timeout=None)

def get_closest_station(lat, lon, weather_data):
    closest_dist = -1
    closest_station = None
    
    for row in weather_data.itertuples():
        dist = distance.distance([lat, lon], [row.Geogr_Breite, row.Geogr_Laenge]).km
        if closest_dist == -1 or dist < closest_dist:
            closest_dist = dist
            closest_station = row
    
    return closest_station

print('Reading data')
data_nds_wetter = reader.get_weather_data()
data_covid = reader.get_covid_data()
print('Data read.') 

covid_nds = data_covid[data_covid.Bundesland.eq('Niedersachsen')]

i = 0
percent = 0
naechste_Station = []
for index, row in covid_nds.iterrows():
    station = get_closest_station(row['Landkreis_Lat'], row['Landkreis_Lon'], data_nds_wetter)
    naechste_Station.append(station)
    i += 1
    if ((i / len(covid_nds)) * 100) >= percent:
        print(str(percent) + '% finished')
        percent += 5

covid_nds['naechste_Station'] = naechste_Station
print(covid_nds.head())            
            
#for row in covid_nds.itertuples():
    

#ListeCovid.landkreis.lat
#ListeCovid.landkreis.lon

#ListeWetter.Station.lat
#ListeWetter.Station.lon

#ListeCovid erweitern um Wetterinformationsspalten
#foreach eintrag in ListeCovid
    #geo_inf_lk = geopy.location(eintrag['Landkreis'])
    #Wetterinfos = coole_funktion(geo_inf_lk, ListeWetter)
    #ListerCovid[Wetterspalten] = Wetterinfos


"""
df_list = []

for row in data_nds_wetter.itertuples():
    datum = row.Zeitstempel
    zeitstempel = dt.datetime.strptime(str(datum), '%Y%m%d').strftime('%d/%m/%Y')
    df_list.append([row.SDO_ID, row.SDO_Name, zeitstempel, row.Wert])
    
data_nds = pd.DataFrame(df_list, columns=['SDO_ID', 'Messstation', 'Datum', 'Temperatur'])

messstation = data_nds['Messstation']=='Diepholz'
data_messstation = data_nds[messstation]

plt.plot(data_messstation.Datum, data_messstation.Temperatur)
plt.title('Temperaturverlauf in Oldenburg')
plt.xlabel('Datum')
plt.ylabel('Temperatur in °C')
plt.show()
"""