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
                    landkreis_Before = rkiDaten[2]
                    anzahl_Faelle += int(rkiDaten[6])
                elif landkreis_Before != rkiDaten[2]:
                    landkreise_Namen.append(landkreis_Before)
                    faelle.append (anzahl_Faelle)
                    anzahl_Faelle = 0
                    landkreis_Before = rkiDaten[2]
                
        landkreise_Namen.append(landkreis_Before)
        anzahl_Faelle += int(rkiDaten[6])
        faelle.append (anzahl_Faelle) 
        
        for landkreis in landkreise_Namen:
            landkreis = landkreis.replace('Ã¼','ü')
       #     location = geolocator.geocode(landkreis)
        #    longitudes.append(location.longitude)
         #   latitudes.append(location.latitude)
            
        
        max_amount = float(10000)

        hmap = folium.Map(location=[42.5, -75.5], zoom_start=7, )
        data = ( 
            np.random.normal(size=(100, 3)) *
            np.array([[1,1,1]]) +
            np.array([[48,5,1]])
            
            ).tolist()
        
        data = []
        for x in liste:
            data.append([x.landkreis.lat, x.landkreis.lng, x.anz_faelle])

  #      hm_wide = HeatMap( list(zip(latitudes.values, longitudes.values, faelle.values)),
   #                       min_opacity=0.2,
    #                      max_val=max_amount,
     #                     radius=17, blur=15, 
      #                    max_zoom=1, 
       #                   )

#        folium.GeoJson(district23).add_to(hmap)
        Heatmap(data).add_to(hmap)
        hmap.save("Result\\" + "test" + ".html")