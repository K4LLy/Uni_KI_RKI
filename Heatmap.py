import folium
import csv
from geopy.geocoders import Nominatim
geolocator = Nominatim(timeout = None)

def generate(covid_data, file_name = "Heatmap"):
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

