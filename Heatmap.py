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
        bundesland_Before = 'null'
        anzahl_Faelle = 0
        anzahl_TodesFaelle = 0
        anzahl_genesen = 0
        bundeslaender_Namen = []
        faelle = []
        todesfaelle = []
        gesund = []
        
        for row in csv_reader_object:
            rkiDaten = row
            allRkiData.append (row)
            if rkiDaten[2] != 'Bundesland':
                if bundesland_Before == rkiDaten[2]:
                    anzahl_Faelle += int(rkiDaten[6])
                    anzahl_TodesFaelle += int(rkiDaten[7])
                    anzahl_genesen += int(rkiDaten[15])
                elif bundesland_Before == 'null':
                    bundesland_Before = rkiDaten[2]
                    anzahl_Faelle += int(rkiDaten[6])
                    anzahl_TodesFaelle += int(rkiDaten[7])
                    anzahl_genesen += int(rkiDaten[15])
                elif bundesland_Before != rkiDaten[2]:
                    bundeslaender_Namen.append(bundesland_Before)
                    faelle.append (anzahl_Faelle)
                    todesfaelle.append (anzahl_TodesFaelle)
                    gesund.append (anzahl_genesen)
                    anzahl_Faelle = 0
                    anzahl_TodesFaelle = 0
                    anzahl_genesen = 0
                    bundesland_Before = rkiDaten[2]
                
        bundeslaender_Namen.append(bundesland_Before)
        anzahl_Faelle += int(rkiDaten[6])
        anzahl_TodesFaelle += int(rkiDaten[7])
        anzahl_genesen += int(rkiDaten[15])
        faelle.append (anzahl_Faelle)
        todesfaelle.append (anzahl_TodesFaelle)
        gesund.append (anzahl_genesen)            
    #Karte mit Fokus auf Deutschland 
    folium_map = folium.Map(location=[51.144, 9.902],
                            zoom_start=6.5,
                            tiles="CartoDB positron")
   

    #pro Bundesland Kreis zeichnen
    index = 0
    for bundesland in bundeslaender_Namen:
        bundesland = bundesland.replace('Ã¼','ü')
        anzahl_faelle = faelle[index]
        anzahl_todesfaelle = todesfaelle[index]
        anzahl_genesen = gesund[index]
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
        index += 1

    #Speichern in HTML
    folium_map.save("Result\\" + file_name + ".html")