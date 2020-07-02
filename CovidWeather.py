#Nina Renken
#Quelle für Wetterdaten:
#https://cdc.dwd.de/portal/201912031600/mapview

import DataReader as reader
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D #wird für projection='3d' gebraucht

def get_covid_weather(bundesland, covid_data):
    data_weather = reader.get_weather_data(bundesland)
    data_weather = data_weather.groupby('Zeitstempel').mean()
    #data_covid   = covid_data
    data_covid   = covid_data[covid_data.Bundesland.eq(bundesland)]
    data_covid   = data_covid.groupby('Meldedatum').sum()
    data_covid   = data_covid.reset_index()
    
    # Präparieren der Wetterdaten
    df_list = []
    for row in data_weather.itertuples():
        datum       = row.Index
        zeitstempel = datetime.strptime(str(datum), '%Y%m%d')
        day_of_year = zeitstempel.timetuple().tm_yday
        zeitstempel = dt.datetime.strptime(str(datum), '%Y%m%d').strftime('%Y/%m/%d')
        df_list.append([zeitstempel, row.Wert, day_of_year])
    
    df_weather = pd.DataFrame(df_list, columns=['Datum', 'Temperatur', 'Tagnr'])
    
    data_covid_weather = pd.merge(data_covid, df_weather, left_on='Meldedatum', right_on='Datum')
    
    
    #Diagramm mit zwei y-Achsen
    fig2y, axleft =plt.subplots()
    
    color = 'tab:red'
    axleft.set_xlabel('Tag Nummer')
    axleft.set_ylabel('Anzahl der Fälle', color=color)
    axleft.plot(data_covid_weather.Tagnr, data_covid_weather.AnzahlFall, color=color)
    axleft.tick_params(axis='y', labelcolor=color)
    axleft.tick_params(axis='x', rotation=90)
    axleft.set_title('Anzahl der Fälle und Temperatur in ' + bundesland)
    
    axright = axleft.twinx()
    
    color = 'tab:blue'
    axright.set_ylabel('Temperatur (°C)', color = color)
    axright.plot(data_covid_weather.Tagnr, data_covid_weather.Temperatur, color=color)
    axright.tick_params(axis='y', labelcolor=color)
    
    fig2y.tight_layout()
    plt.show()
    
    
    #Erstellen des 3D-Koordinatensystems mit Scatterplots
    figScatter = plt.figure()
    axScatter = figScatter.add_subplot(111, projection='3d')
    axScatter.scatter(data_covid_weather.Tagnr, data_covid_weather.Temperatur, data_covid_weather.AnzahlFall, 
                      c=np.linalg.norm([data_covid_weather.Tagnr, data_covid_weather.AnzahlFall, data_covid_weather.Temperatur], axis=0))
    axScatter.set_xlabel('Tag Nummer')
    axScatter.set_ylabel('Temperatur')
    axScatter.set_zlabel('Anzahl Fälle')
    axScatter.set_title('Scatterplot: Datum - Temperatur - Anzahl Fälle in ' + bundesland)
    
    
    #Erstellen des 3D-Koordinatensystems mit trisurf
    figTrisurf = plt.figure()
    axTrisurf = plt.axes(projection='3d')
    axTrisurf.plot_trisurf(data_covid_weather.Tagnr, data_covid_weather.Temperatur, data_covid_weather.AnzahlFall, 
                           cmap='viridis', edgecolor='none')
    axTrisurf.set_xlabel('Tag Nummer')
    axTrisurf.set_ylabel('Temperatur')
    axTrisurf.set_zlabel('Anzahl Fälle')
    axTrisurf.set_title('Trisurf: Datum - Temperatur - Anzahl Fälle in ' + bundesland)