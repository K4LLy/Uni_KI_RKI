import pandas as pd

def getCovidData():
    return pd.read_csv('Data\\RKI_COVID19.csv')
    
def getBundeslaender():
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
    
def getLandkreise(): #Vermutlich müssen hier irgendwann noch die Spaltennamen angepasst werden.
    return pd.read_csv('Data\\landkreise-in-germany.csv', sep=';')

def combine(df_left, df_right, key):
    return pd.merge(df_left, df_right, on=key)