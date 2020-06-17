import matplotlib.pyplot as matplot
import datetime as dt

#Kuchendiagramm
def generate_pie_chart(covid_data):
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
        explode.append(0)
         
        
    fig1, ax1 = matplot.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    matplot.show()  

#Balkendiagramm
def generate_bar_chart(covid_data):
    data = covid_data.groupby(['Geschlecht']).sum()
    
    counts = []
    header = []
    
    for geschlecht, row in data.iterrows():
        counts.append(row['AnzahlFall'])
        header.append(geschlecht)
        
    matplot.bar(header, counts, color = 'red')
    matplot.title("Männlich | Weiblich")
    matplot.show()
    
#Grafik nach Meldedatum
def generate_graph(covid_data):
    data = covid_data.sort_values(by = ['Meldedatum']).groupby(['Meldedatum']).sum()
    
    dates_to_plot = []
    count_dates_to_plot = []
    
    for date, row in data.iterrows():
        dates_to_plot.append(date.replace(' 00:00:00', ''))
        count_dates_to_plot.append(row['AnzahlFall'])
    
    matplot.plot(dates_to_plot, count_dates_to_plot)