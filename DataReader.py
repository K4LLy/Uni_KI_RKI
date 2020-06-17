import pandas as pd
    
def get_bundeslaender_data():
    bl_data = pd.read_csv('Data\\bundesland.csv', sep=';')
    bl_data.rename(columns={'State name': 'Bundesland',
                            'Geo Point': 'Geo_Point',
                            'Geo Shape': 'Geo_Shape',
                            'State Code': 'Bundesland_Code',
                            'State Type': 'Bundesland_Typ',
                            'NUTS Code': 'Nuts_Code',
                            'Population': 'Einwohnerzahl'},
                   inplace=True)
    return bl_data
    
def get_landkreise_data(): #Vermutlich müssen hier irgendwann noch die Spaltennamen angepasst werden.
    return pd.read_csv('Data\\landkreise-in-germany.csv', sep=';')

def combine(df_left, df_right, key):
    return pd.merge(df_left, df_right, on=key)

def get_covid_data():
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
            
        lk_name = row['Landkreis'].replace('LK ', '').replace('SK ', '').replace('StadtRegion ', '')
        df_list.append([row['Bundesland'], lk_name, anzahlFall, anzahlGenesen, anzahlTodesfall,
                        row['Altersgruppe'], row['Geschlecht'], row['Meldedatum'].replace(' 00:00:00', '')])
        
    return pd.DataFrame(df_list, columns=['Bundesland', 'Landkreis', 'AnzahlFall', 'AnzahlGenesen', 
                                          'AnzahlTodesfall', 'Altersgruppe', 'Geschlecht',
                                          'Meldedatum'])