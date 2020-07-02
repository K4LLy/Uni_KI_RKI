#Pascal Winkler
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
        printProgressBar(index, (len(data) - 1))
        
    print('Covid Data read.')
    return pd.DataFrame(df_list, columns=['Bundesland', 'Landkreis', 'AnzahlFall', 'AnzahlGenesen', 
                                          'AnzahlTodesfall', 'Altersgruppe', 'Geschlecht',
                                          'Meldedatum', 'Kalenderwoche'])

def get_weather_data(bundesland):
    data_station    = pd.read_csv('Data\\Wetterdaten\\sdo_' + bundesland + '.csv')
    data_temperatur = pd.read_csv('Data\\Wetterdaten\\data_' + bundesland + '.csv')
    return(combine(data_temperatur, data_station, 'SDO_ID'))

#https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()