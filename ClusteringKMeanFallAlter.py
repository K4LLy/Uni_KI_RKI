

import DataReader as reader
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pandas import DataFrame
covid_data = reader.get_covid_data()


                                       

altergruppe_durchschnittsalter = []
altergruppe = ''
indexes = []
for index, row in covid_data.iterrows():
    altergruppe = row['Altersgruppe']
    if altergruppe == 'A00-A04':
        altergruppe_durchschnittsalter.append(2)
    elif altergruppe == 'A05-A14':
        altergruppe_durchschnittsalter.append(10)
    elif altergruppe == 'A15-A34':
        altergruppe_durchschnittsalter.append(25)
    elif altergruppe == 'A35-A59':
        altergruppe_durchschnittsalter.append(47)
    elif altergruppe == 'A60-A79':
        altergruppe_durchschnittsalter.append(70)
    elif altergruppe == 'A80+':
        altergruppe_durchschnittsalter.append(80)
    else:
       altergruppe_durchschnittsalter.append(0) 
    indexes.append(1)
 
covid_data.loc[:,('Altersgruppedurchschnitt')] = altergruppe_durchschnittsalter
covid_data.loc[:,('Index')] = indexes
dataframe = covid_data.groupby(['Bundesland', 'Altersgruppe']).sum()
for index, row in dataframe.iterrows():
    altergruppe = row['Altersgruppedurchschnitt']
    anzahl = row['Index']
    dataframe.at[index, 'Altersgruppedurchschnitt'] = altergruppe/anzahl

        
dataframe = dataframe.filter(items= ['Altersgruppedurchschnitt', 'AnzahlFall'])


    
kmeans = KMeans(n_clusters=10).fit(dataframe)
centroids = kmeans.cluster_centers_
print(centroids)

plt.scatter(dataframe['Altersgruppedurchschnitt'], dataframe['AnzahlFall'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.title('Anzahl Fall pro Altersgruppe (Durschnitt) pro Bundesland')
plt.xlabel('Altersgruppe Durschnittsalter')
plt.ylabel('Fälle')

plt.show()