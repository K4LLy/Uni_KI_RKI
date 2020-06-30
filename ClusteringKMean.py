import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
def cluster_kmean(covid_data):
    dataframe = covid_data.groupby(['Bundesland', 'Altersgruppe']).sum()
    dataframe = dataframe.filter(items= ['AnzahlFall', 'AnzahlTodesfall'])
    kmeans = KMeans(n_clusters=5).fit(dataframe)
    centroids = kmeans.cluster_centers_
    print(centroids)

    plt.scatter(dataframe['AnzahlFall'], dataframe['AnzahlTodesfall'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
    plt.title('Anzahl Fall und Todesfall pro Bundesland pro Altergruppe')
    plt.xlabel('Fälle')
    plt.ylabel('davon Todesfälle')

    
    plt.savefig("Result\\" + 'clustering_k_mean_faelle_todesfaelle_bundesland_altersgruppe' + ".png")
    plt.show()