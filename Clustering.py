"""
Lara Ahrens
"""
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def cluster_kmean_fall_alter(covid_data):                                     
    print('Create Cluster Fall Alter...')
    
    data = covid_data.copy()
    altergruppe_durchschnittsalter = []
    altergruppe = ''
    indexes = []
    for index, row in data.iterrows():
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
 
    data.loc[:,('Altersgruppedurchschnitt')] = altergruppe_durchschnittsalter
    data.loc[:,('Index')] = indexes
    data = data.groupby(['Bundesland', 'Altersgruppe']).sum()
    for index, row in data.iterrows():
        altergruppe = row['Altersgruppedurchschnitt']
        anzahl = row['Index']
        data.at[index, 'Altersgruppedurchschnitt'] = altergruppe/anzahl
        
    data = data.filter(items= ['Altersgruppedurchschnitt', 'AnzahlFall'])
    
    kmeans = KMeans(n_clusters=10).fit(data)
    centroids = kmeans.cluster_centers_
    print(centroids)

    plt.scatter(data['Altersgruppedurchschnitt'], data['AnzahlFall'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
    plt.title('Anzahl Fall pro Altersgruppe (Durschnitt) pro Bundesland')
    plt.xlabel('Altersgruppe Durschnittsalter')
    plt.ylabel('Fälle')

    
    plt.savefig("Result\\" + 'clustering_k_mean_fall_alter' + ".png")
    plt.show()
    print('Cluster created.')
    
def cluster_kmean_faelle_todesfaelle(covid_data):
    print('Create Cluster Fall Todesfall...')
    data = covid_data.groupby(['Bundesland', 'Altersgruppe']).sum()
    data = data.filter(items= ['AnzahlFall', 'AnzahlTodesfall'])
    kmeans = KMeans(n_clusters=5).fit(data)
    centroids = kmeans.cluster_centers_
    print(centroids)

    plt.scatter(data['AnzahlFall'], data['AnzahlTodesfall'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
    plt.title('Anzahl Fall und Todesfall pro Bundesland pro Altergruppe')
    plt.xlabel('Fälle')
    plt.ylabel('davon Todesfälle')
    print('Cluster created.')
    
    plt.savefig("Result\\" + 'clustering_k_mean_faelle_todesfaelle_bundesland_altersgruppe' + ".png")
    plt.show()