import pandas as pd
import datetime as dt
from geopy import Nominatim
geolocator = Nominatim(timeout=None)

def combine(df_left, df_right, key):
    return pd.merge(df_left, df_right, on=key)

def get_covid_data():
    print('Reading Covid Data...')
    data = pd.read_csv('Data\\RKI_COVID19.csv')
    
    df_list = []    
    cache_geoloc = {}
    data_count = len(data)
    geocode = lambda query: geolocator.geocode("%s, Deutschland" % query)
    i = 0
    percent = 0
    
    for row in data.itertuples():
        i += 1
        if ((i / data_count) * 100) >= percent:
            print(str(percent) + '% finished')
            percent += 5
        
        neuerFall = row.NeuerFall
        neuerTodesfall = row.NeuerTodesfall
        neuGenesen = row.NeuGenesen
        
        anzahlFall = row.AnzahlFall
        anzahlTodesfall = row.AnzahlTodesfall
        anzahlGenesen = row.AnzahlGenesen
        
        if neuerFall == -1:
            anzahlFall = 0
            
        if neuerTodesfall == -1:
            anzahlTodesfall = 0
            
        if neuGenesen == -1:
            anzahlGenesen = 0
            
        meldedatum = row.Meldedatum.replace(' 00:00:00', '')
        kw = dt.datetime.strptime(meldedatum, '%Y/%m/%d').strftime('%W')
        lk_name = row.Landkreis.replace('LK ', '').replace('SK ', '').replace('StadtRegion ', '')
                    
        loc_lk = None
        if lk_name in cache_geoloc:
            loc_lk = cache_geoloc[lk_name]
        else:
            loc_lk = geocode(lk_name)
            cache_geoloc[lk_name] = loc_lk
            
        loc_bl = None
        if row.Bundesland in cache_geoloc:
            loc_bl = cache_geoloc[row.Bundesland]
        else:
            loc_bl = geocode(row.Bundesland)
            cache_geoloc[row.Bundesland] = loc_bl
        
        df_list.append([row.Bundesland, lk_name, anzahlFall, anzahlGenesen, anzahlTodesfall,
                        row.Altersgruppe, row.Geschlecht, meldedatum, kw, loc_lk.latitude, loc_lk.longitude,
                        loc_bl.latitude, loc_bl.longitude])
        
    print('Covid Data read.')
    return pd.DataFrame(df_list, columns=['Bundesland', 'Landkreis', 'AnzahlFall', 'AnzahlGenesen', 
                                          'AnzahlTodesfall', 'Altersgruppe', 'Geschlecht',
                                          'Meldedatum', 'Kalenderwoche', 'Landkreis_Lat', 'Landkreis_Lon',
                                          'Bundesland_Lat', 'Bundesland_Lon'])

def get_weather_data():
    data_lufttemp = pd.read_csv('Data\\Wetterdaten\\mittlere_Lufttemperatur\\data\\data_TMK_MN004.csv')
    data_stationen = pd.read_csv('Data\\Wetterdaten\\mittlere_Lufttemperatur\\data\\sdo_TMK_MN004.csv')
    return combine(data_lufttemp, data_stationen, 'SDO_ID')