import matplotlib.pyplot as matplot
import datetime as dt

#Kuchendiagramm
def generate_pie_chart(covid_data):
    covid_data['Altersgruppe'].replace({'A05-A14' : 'Sonstige', 'A00-A04' : 'Sonstige', 'unbekannt' : 'Sonstige', 
                                       'A80+' : '80+ J.', 'A60-A79' : '60-79 J.', 'A35-A59' : '35-59 J.', 'A15-A34' : '15-34 J.',  }, inplace=True)
    data = covid_data.groupby(['Altersgruppe']).sum()
    
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
    
    
  #  ax1.legend(data, labels,
   #       title="Altersgruppen",
    #      loc="center left",
     #     bbox_to_anchor=(1, 0, 0.5, 1))

    #matplot.setp(autotexts, size=8, weight="bold")

    ax1.set_title("Erkrankungen nach Altersgruppen:")
    
    
    
    matplot.show()  




#Balkendiagramm
def generate_bar_chart(covid_data):
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
    matplot.show()
    
    
    
    
#Grafik nach Meldedatum
def generate_graph(covid_data):
    for index, row in covid_data.iterrows():
        kw = dt.datetime.strptime(row['Meldedatum'], '%Y/%m/%d').strftime('%W')
        row['Meldedatum'] = str(kw)
    
    data = covid_data.sort_values(by = ['Meldedatum']).groupby(['Meldedatum']).sum()
    
    dates_to_plot = []
    count_dates_to_plot = []
    
    for date, row in data.iterrows():
        dates_to_plot.append(date)
        count_dates_to_plot.append(row['AnzahlFall'])
    
    matplot.plot(dates_to_plot, count_dates_to_plot)