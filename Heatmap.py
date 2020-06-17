import folium
import csv
from geopy.geocoders import Nominatim
geolocator = Nominatim(timeout = None)

def generate(file_name = "Heatmap"):    

    with open('Data\\RKI_COVID19.csv') as csvdatei:
        csv_reader_object = csv.reader(csvdatei, delimiter=',')
    
        bundeslaender = []
        allRkiData = []
        rkiDaten = []
        bundeslandBefore = 'null'
        anzahlFaelle = 0
        anzahlTodesFaelle = 0
        anzahlGenesen = 0
        bundeslaenderNamen = []
        faelle = []
        todesfaelle = []
        gesund = []
        
        for row in csv_reader_object:
            rkiDaten = row
            allRkiData.append (row)
            if rkiDaten[2] != 'Bundesland':
                if bundeslandBefore == rkiDaten[2]:
                    anzahlFaelle += int(rkiDaten[6])
                    anzahlTodesFaelle += int(rkiDaten[7])
                    anzahlGenesen += int(rkiDaten[15])
                elif bundeslandBefore == 'null':
                    bundeslandBefore = rkiDaten[2]
                    anzahlFaelle += int(rkiDaten[6])
                    anzahlTodesFaelle += int(rkiDaten[7])
                    anzahlGenesen += int(rkiDaten[15])
                elif bundeslandBefore != rkiDaten[2]:
                    bundeslaenderNamen.append(bundeslandBefore)
                    faelle.append (anzahlFaelle)
                    todesfaelle.append (anzahlTodesFaelle)
                    gesund.append (anzahlGenesen)
                    anzahlFaelle = 0
                    anzahlTodesFaelle = 0
                    anzahlGenesen = 0
                    bundeslandBefore = rkiDaten[2]
                
        bundeslaenderNamen.append(bundeslandBefore)
        anzahlFaelle += int(rkiDaten[6])
        anzahlTodesFaelle += int(rkiDaten[7])
        anzahlGenesen += int(rkiDaten[15])
        faelle.append (anzahlFaelle)
        todesfaelle.append (anzahlTodesFaelle)
        gesund.append (anzahlGenesen)            
    #Karte mit Fokus auf Deutschland 
    folium_map = folium.Map(location=[51.144, 9.902],
                            zoom_start=6.5,
                            tiles="CartoDB positron")
   

    #pro Bundesland Kreis zeichnen
    index = 0
    for bundesland in bundeslaenderNamen:
        bundesland = bundesland.replace('Ã¼','ü')
        anzahl_faelle = faelle[index]
        anzahl_todesfaelle = todesfaelle[index]
        anzahl_genesen = gesund[index]
        anzeige_string = str(anzahl_faelle) + " Fälle insgesamt in " + bundesland
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
        index += 1

    #Speichern in HTML
    folium_map.save("Result\\" + file_name + ".html")