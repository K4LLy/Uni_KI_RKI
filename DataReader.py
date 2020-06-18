import pandas as pd
import datetime as dt

def combine(df_left, df_right, key):
    return pd.merge(df_left, df_right, on=key)

def get_covid_data():
    print('Reading Covid Data...')
    data = pd.read_csv('Data\\RKI_COVID19.csv')
    
    df_list = []
    
    for index, row in data.iterrows():
        neuerFall = row['NeuerFall']
        neuerTodesfall = row['NeuerTodesfall']
        neuGenesen = row['NeuGenesen']
        
        anzahlFall = row['AnzahlFall']
        anzahlTodesfall = row['AnzahlTodesfall']
        anzahlGenesen = row['AnzahlGenesen']
        
        if neuerFall == -1:
            anzahlFall = 0
            
        if neuerTodesfall == -1:
            anzahlTodesfall = 0
            
        if neuGenesen == -1:
            anzahlGenesen = 0
            
        meldedatum = row['Meldedatum'].replace(' 00:00:00', '')
        kw = dt.datetime.strptime(meldedatum, '%Y/%m/%d').strftime('%W')
        lk_name = row['Landkreis'].replace('LK ', '').replace('SK ', '').replace('StadtRegion ', '')
        df_list.append([row['Bundesland'], lk_name, anzahlFall, anzahlGenesen, anzahlTodesfall,
                        row['Altersgruppe'], row['Geschlecht'], meldedatum, kw])
        
    print('Covid Data read.')
    return pd.DataFrame(df_list, columns=['Bundesland', 'Landkreis', 'AnzahlFall', 'AnzahlGenesen', 
                                          'AnzahlTodesfall', 'Altersgruppe', 'Geschlecht',
                                          'Meldedatum', 'Kalenderwoche'])