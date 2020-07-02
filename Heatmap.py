#Auslagern der 3 Dateien in eine und Refactoring: Pascal
#Urheber der Inhalte der Funktionen: in der jeweiligen Funktion
import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from folium.plugins import HeatMap as Heatmap
from geopy.geocoders import Nominatim
geolocator = Nominatim(timeout = None)

def generate_circle(covid_data, file_name = "Circlemap"):
    #Lara Ahrens
    print('Generate Circlemap...')
    data = covid_data.groupby(['Bundesland']).sum()

    #Karte mit Fokus auf Deutschland 
    folium_map = folium.Map(location=[51.144, 9.902],
                            zoom_start=6.5,
                            tiles="CartoDB positron")

    #pro Bundesland Kreis zeichnen
    for bundesland, row in data.iterrows():
        anzahl_faelle = row['AnzahlFall']
        anzahl_todesfaelle = row['AnzahlTodesfall']
        anzahl_genesen = row['AnzahlGenesen']
        anzeige_string = str(anzahl_faelle) + " Fälle insgesamt, " + str(anzahl_genesen) + " gesunde Fälle und "+ str(anzahl_todesfaelle) + " Todesfälle in " + bundesland
        location = geolocator.geocode(bundesland) #mit dem Geolocator können die Koordinaten von dem Bundesland hergeholt werden
        folium.Circle(
            location = [location.latitude, location.longitude], #Kreis liegt auf dem Zentrum des Bundeslandes
            popup = anzeige_string,
            radius = float(anzahl_faelle) * 2, #Radius ist von den Fallzahlen abhängig
            color = 'red',
            fill = True,
            fill_color = 'red'
            ).add_to(folium_map) #hinzufügen der Kreise auf die Karte
        folium.Circle(
            location = [location.latitude, location.longitude], #Kreis liegt auf dem Zentrum des Bundeslandes
            popup = anzeige_string,
            radius = float(anzahl_genesen) * 2, #Radius ist von den Fallzahlen abhängig
            color = 'green',
            fill = True,
            fill_color = 'green'
            ).add_to(folium_map) #hinzufügen der Kreise auf die Karte
        folium.Circle(
            location = [location.latitude, location.longitude], #Kreis liegt auf dem Zentrum des Bundeslandes
            popup = anzeige_string,
            radius = float(anzahl_todesfaelle) * 2, #Radius ist von den Fallzahlen abhängig
            color = 'black',
            fill = True,
            fill_color = 'black'
            ).add_to(folium_map) #hinzufügen der Kreise auf die Karte

    #Speichern in HTML
    folium_map.save("Result\\" + file_name + ".html")
    print('Circlemap created.')

def generate_heatmap(covid_data, file_name = "Heatmap"):
    #Grundstruktur: Lara Ahrens und Gerriet Schmidt
    #Refactoring Pascal Winkler
    print('Generate Heatmap...')
    covid_grouped = covid_data.groupby(['Landkreis']).sum()
    
    heatmap_data = []
    for lk, row in covid_grouped.iterrows():
        geocode = lambda query: geolocator.geocode("%s, Deutschland" % query)
        lk_location = geocode(lk)
        
        if lk_location != None:
            heatmap_data.append([lk_location.latitude, lk_location.longitude, int(row['AnzahlFall'])])
            
        
    hmap = folium.Map(location=[51.144, 9.902],
                            zoom_start=6.5,
                            tiles="CartoDB positron")

    Heatmap(heatmap_data).add_to(hmap)
    hmap.save("Result\\" + file_name + ".html")
    print('Heatmap created.')
    
def generate_chart(covid_data):
    #Nina Renken
    print('Creating Heatmapchart...')
    data = covid_data.groupby(['Altersgruppe']).sum()
    
    array = []
    label_status = ["Krank", "Genesen", "Gestorben"]
    label_altersgruppe = []
    
    for altersgruppe, row in data.iterrows():
        label_altersgruppe.append(altersgruppe)
        array.append([int(row['AnzahlFall']), int(row['AnzahlGenesen']), int(row['AnzahlTodesfall'])])
    
    covid_alter_abhaengigkeit = np.array(array)
    
    fig, ax = plt.subplots()
    im = ax.imshow(covid_alter_abhaengigkeit)
    
    #Erstellen der Achsen + Beschriftung und Rotation dessen
    ax.set_xticks(np.arange(len(label_status)))
    ax.set_yticks(np.arange(len(label_altersgruppe)))
    
    ax.set_xticklabels(label_status)
    ax.set_yticklabels(label_altersgruppe)
    ax.set_title('Heatmap: Altersgruppen - erkrankt, genesen, tot')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    #Zellenbeschriftung
    for i in range(len(label_altersgruppe)):
        for j in range(len(label_status)):
            text = ax.text(j, i, covid_alter_abhaengigkeit[i, j], ha="center", va="center", color="w")
    
    fig.tight_layout()
    plt.show()
    print('Heatmapchart created.')