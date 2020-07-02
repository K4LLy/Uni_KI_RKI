#Nina Renken
#Quelle für Koordinaten der 5G-Masten
#https://www.5g-anbieter.info/verfuegbarkeit/5g-verfuegbarkeit-testen.html

import pandas as pd
import folium
import DataReader as reader
from folium.plugins import HeatMap as Heatmap
from geopy.geocoders import Nominatim
geolocator = Nominatim(timeout = None)
   
#der Heatmap-Part ist aus Heatmap kopiert
print('Generate Heatmap...')
covid_data = reader.get_covid_data()
covid_grouped = covid_data.groupby(['Landkreis']).sum()

heatmap_data = []
for lk, row in covid_grouped.iterrows():
    geocode = lambda query: geolocator.geocode("%s, Deutschland" % query)
    lk_location = geocode(lk)
    
    if lk_location != None:
        heatmap_data.append([lk_location.latitude, lk_location.longitude, int(row['AnzahlFall'])])
        
    
folium_map = folium.Map(location=[51.144, 9.902],
                        zoom_start=6.5,
                        tiles="CartoDB positron")

Heatmap(heatmap_data).add_to(folium_map)
print('Generated Heatmap.')


data = pd.read_csv('Data\\5G_Masten_Koordinaten.csv')
data = data.drop(columns=['Unnamed: 0'])

for row in data.itertuples():
   folium.Marker([row.Geogr_Breite, row.Geogr_Laenge]).add_to(folium_map)

folium_map.save('Heatmap+5G.html')