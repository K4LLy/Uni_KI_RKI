
import DataReader as reader
covid_data = reader.getCovidData()
dataframe = covid_data.groupby(['Meldedatum']).sum()

dataframe = dataframe.filter(items=['AnzahlGenesen'])
for i in range(5):
    i = i +6
    newdata = dataframe.shift(i, axis = 0)
    dataframe.loc[:,('Genes-'+str(i))] = newdata['AnzahlGenesen']
    
dataframe.dropna(inplace=True)    

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#lineares Modell
features = np.array(dataframe.drop(['AnzahlGenesen'],1))
labels = np.array(dataframe.filter(items=['AnzahlGenesen']))
features = preprocessing.scale(features)
#Test- und Trainingsdaten
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

linear_classifier = LinearRegression()
linear_classifier.fit(features_train, labels_train)
score = linear_classifier.score(features_test,labels_test)
print(score)


    






