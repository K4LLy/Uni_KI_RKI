#alle
import DataReader as reader
import Heatmap as hm
import Charts as ch
import LinReg as lr
import NeuralNetwork as nn
import Util as u
import Clustering as cluster
import RegressionTreeAnzFaelle as rt
import bundeslaender_knn_linReg_linBaum as bundeslaender
#import CovidWeather as cw
#import Funkmasten as fm
import QuellcodeGY as gy

from IPython import get_ipython

#get_ipython().run_line_magic('matplotlib', 'qt') - Grafik in extra Fenster
#get_ipython().run_line_magic('matplotlib', 'inline') - Grafik in Spyder

get_ipython().run_line_magic('matplotlib', 'inline')

covid_data = reader.get_covid_data()

pred_col = 'AnzahlFall'
test_name = 'P_Test1'
offset = 15
shown_days = 60
pred_days = 20

bundesland = 'Niedersachsen'
kalenderwoche = 27
grossveranstaltung = 1 #ja
maskenpflicht = 1 #ja
kontaktbeschraenkung = 1 #ja

X, X_pred, y, data = u.prepare_data(covid_data, pred_col, offset, pred_days)
X_train, X_test, y_train, y_test = u.get_train_test(X, y, 0.2)


print('Press Enter to generate the graph...')
carryon = input()
ch.generate_graph(covid_data)

print('Press Enter to generate the pie chart...')
carryon = input()
ch.generate_pie_chart(covid_data)

print('Press Enter to generate the bar chart...')
carryon = input()
ch.generate_bar_chart(covid_data)


print('Press Enter to generate the heatmap chart...')
carryon = input()
hm.generate_chart(covid_data)

print('Press Enter to generate the heatmap...')
carryon = input()
#hm.generate_heatmap(covid_data)

print('Press Enter to generate the map with circles...')
carryon = input()
#hm.generate_circle(covid_data)


print('Press Enter to generate charts for Covid-19 and weather ...')
carryon = input()
#cw.get_covid_weather(bundesland, covid_data)

print('Press Enter to generate the heatmap with markers for 5G masts ...')
carryon = input()
#fm.generate_heatmap_5g(covid_data, 'Heatmap_5G')


print('Press Enter to start linear regression...')
carryon = input()
gy.predict_old(data, offset, len(X_test), shown_days, pred_col, print_metric = True)

#lr.create(X_train, y_train, test_name, save_as_file = True)
linear_regression = lr.load(test_name)
y_pred = lr.predict(linear_regression, data, X_pred, pred_col, shown_days)
lr.print_info(linear_regression, X_test, y_test, y[-len(y_pred):], y_pred)



print('Press Enter to continue clustering...')
carryon = input()
cluster.cluster_kmean_faelle_todesfaelle(covid_data)

print('Press Enter to continue clustering...')
carryon = input()
cluster.cluster_kmean_fall_alter(covid_data)


print('Press Enter to continue with the regression tree...')
carryon = input()
rt.regressionTree(covid_data)


print('Press Enter to train the neural network ...')
carryon = input()
#nn.create(X_train, y_train, unique_name = test_name, save_as_file = True)
neural_network = nn.load(test_name)
nn.predict(neural_network, data, X_pred, pred_col, shown_days)
nn.print_info(neural_network, X_test, y_test)


print('Press Enter to continue...')
carryon = input()
bundeslaender.predict_Data_for_onehot_encoded_bundesland(covid_data, kalenderwoche, pred_col, grossveranstaltung, maskenpflicht, kontaktbeschraenkung)

print('Press Enter to continue...')
carryon = input()
bundeslaender.predict_Data_for_one_bundesland(covid_data, kalenderwoche, pred_col, grossveranstaltung, bundesland,maskenpflicht, kontaktbeschraenkung)

print('Press Enter to continue...')
carryon = input()
bundeslaender.predict_data_with_knn_multi_label(covid_data, pred_col, kalenderwoche, grossveranstaltung,maskenpflicht,kontaktbeschraenkung)
