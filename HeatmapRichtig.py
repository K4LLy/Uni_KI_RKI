# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:15:39 2020

@author: gerri
"""


import folium
import csv
from geopy.geocoders import Nominatim

geolocator = Nominatim(timeout = None)



import pandas as pd 
import folium
from folium.plugins import HeatMap as Heatmap
import numpy as np

    
with open('Data\\RKI_COVID19.csv') as csvdatei:
        csv_reader_object = csv.reader(csvdatei, delimiter=',')
    
        landkreise = []
        allRkiData = []
        rkiDaten = []
        landkreis_Before = 'null'
        anzahl_Faelle = 0
        landkreise_Namen = []
        faelle = []
        longitudes = []
        latitudes = []
        
        for row in csv_reader_object:
            rkiDaten = row
            allRkiData.append (row)
            if rkiDaten[3] != 'Landkreis':
                if landkreis_Before == rkiDaten[3]:
                    anzahl_Faelle += int(rkiDaten[6])
                elif landkreis_Before == 'null':
                    landkreis_Before = rkiDaten[3]
                    anzahl_Faelle += int(rkiDaten[6])
                elif landkreis_Before != rkiDaten[3]:
                    landkreise_Namen.append(landkreis_Before)
                    faelle.append (anzahl_Faelle)
                    anzahl_Faelle = 0
                    landkreis_Before = rkiDaten[3]
                
        landkreise_Namen.append(landkreis_Before)
        anzahl_Faelle += int(rkiDaten[6])
        faelle.append (anzahl_Faelle) 
        
        for landkreis in landkreise_Namen:
            landkreis = landkreis.replace('Ã¼','ü')
            landkreis = landkreis.replace('Ã¶', 'ö')
            location = geolocator.geocode(landkreis)
            if location is not None:
                longitudes.append(location.longitude)
                latitudes.append(location.latitude)
            
        
      
        hmap = folium.Map(location=[51.144, 9.902],
                            zoom_start=6.5,
                            tiles="CartoDB positron")
        
        
        data = []
        index = 0
        for x in latitudes:
            data.append([latitudes[index], longitudes[index], faelle[index]])
            index += 1

        Heatmap(data).add_to(hmap)
        hmap.save("Result\\" + "test" + ".html")