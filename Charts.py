import matplotlib.pyplot as matplot
#Einfuegen der einzelnen Dateien in Funktionen und Refactoring: Pascal
#Urheber der Inhalte der Funktionen: in der jeweiligen Funktion
#Kuchendiagramm
def generate_pie_chart(covid_data):
    #Grundstruktur Lara Ahrens
    #Refactor Gerriet Schmidt (Verschoenerung des Graphen, Achsenbeschriftungen)
    print('Creating Piechart...')
    data = covid_data.copy()
    data['Altersgruppe'].replace({'A05-A14' : 'Sonstige', 'A00-A04' : 'Sonstige', 'unbekannt' : 'Sonstige', 
                                  'A80+' : '80+ J.', 'A60-A79' : '60-79 J.', 'A35-A59' : '35-59 J.',
                                  'A15-A34' : '15-34 J.',  }, inplace=True)
    data = data.groupby(['Altersgruppe']).sum()
    
    all_cases = 0
    labels = []
    sizes = []
    explode = []
    
    for entry in data['AnzahlFall']: #Gibt bestimmt noch ne schönere Lösung dafür
        all_cases += entry
    
    for gruppe, row in data.iterrows():
        labels.append(gruppe)
        sizes.append(float(row['AnzahlFall']) / all_cases * 100)
        
        if float(row['AnzahlFall']) > 10000:
            explode.append(0.01) #entfernung zwischen Wedges für das Explode
        else:
            explode.append(0.5)
    
    fig1, ax1 = matplot.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        startangle=90, textprops = {'size':'9', 'color':'black'})
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title("Erkrankungen nach Altersgruppen:")    
    
    matplot.savefig("Result\\" + 'pie_chart_erkrankungen_altersgruppe' + ".png")
    matplot.show() 
    print('Piechart created.')

#Balkendiagramm
def generate_bar_chart(covid_data):
    #Grundstruktur Lara Ahrens
    #Refactor Gerriet Schmidt (Verschoenerung des Graphen, Achsenbeschriftungen)
    print('Creating Barchart...')
    data = covid_data.groupby(['Geschlecht']).sum()
    
    counts = []
    header = []
    
    for geschlecht, row in data.iterrows():
        counts.append(row['AnzahlFall'])
        if geschlecht == 'M':
            geschlecht = 'Männlich'
        elif geschlecht == 'W':
            geschlecht = 'Weiblich'
        else:
            geschlecht = 'Unbekannt'
            
        header.append(geschlecht)
        
    matplot.bar(header, counts, color = ['#0404B4', '#FE9A2E', 'red'])
    matplot.title("Erkrankungen nach Geschlecht")
    
    print('Barchart created.')    
    matplot.savefig("Result\\" + 'barchart_erkrankung_geschlecht' + ".png")
    matplot.show()
#Grafik nach Meldedatum
def generate_graph(covid_data): 
    #Grundstruktur Lara Ahrens mit Graph fuer jedes Meldedatum 
    #Refactor Gerriet Schmidt (Verschoenerung des Graphen, Achsenbeschriftungen)
    print('Creating Graph...') 
    data = covid_data.sort_values(by = ['Kalenderwoche']).groupby(['Kalenderwoche']).sum()
    
    dates_to_plot = []
    count_dates_to_plot = []
    
    for date, row in data.iterrows():
        dates_to_plot.append(date)
        count_dates_to_plot.append(row['AnzahlFall'])
    
    fig1, ax1 = matplot.subplots()
    ax1.set_ylabel('Anzahl Neuerkrankungen')
    ax1.set_xlabel('Kalenderwoche')
    matplot.title("Anzahl Neuerkrankungen pro KW")    
    matplot.plot(dates_to_plot, count_dates_to_plot, color='green')
    print('Graph created.') 
    matplot.savefig("Result\\" + 'graph_anzahl_neuerkrankungen_pro_kw' + ".png")