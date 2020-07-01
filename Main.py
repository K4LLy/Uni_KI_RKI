import DataReader as reader
import Heatmap as hm
import Charts as ch
import LinReg as lr
import NeuralNetwork as nn
import Util as u
import Clustering as cluster
import RegressionTreeAnzFaelle as rt
import bundeslaender_knn_linReg_linBaum as bundeslaender

covid_data = reader.get_covid_data()

ch.generate_pie_chart(covid_data)
ch.generate_bar_chart(covid_data)
ch.generate_graph(covid_data)

hm.generate_chart(covid_data)
hm.generate_circle(covid_data)
hm.generate_heatmap(covid_data)
cluster.cluster_kmean_faelle_todesfaelle(covid_data)
cluster.cluster_kmean_fall_alter(covid_data)


rt.regressionTree(covid_data)
pred_col = 'AnzahlFall'
X, y, data = u.prepare_data(covid_data, pred_col, offset)
X_train, X_test, y_train, y_test = u.get_train_test(X, y, 0.2)

nn.create(X_train, y_train, unique_name = test_name, save_as_file = True)
neural_network = nn.load(test_name)
nn.predict(neural_network, data, X_test, pred_col, shown_days)
nn.print_info(neural_network, X_test, y_test, False)

lr.create(X_train, y_train, test_name, save_as_file = True)
linear_regression = lr.load(test_name)
y_pred = lr.predict(linear_regression, data, X_test, pred_col, shown_days)
lr.print_info(linear_regression, X_test, y_test, y_pred)


test_name = 'P_Test1'

offset = 15
shown_days = 60

X, y, data = u.prepare_data(covid_data, pred_col, offset)
X_train, X_test, y_train, y_test = u.get_train_test(X, y, 0.2)

nn.create(X_train, y_train, unique_name = test_name, save_as_file = True)
neural_network = nn.load(test_name)
nn.predict(neural_network, data, X_test, pred_col, shown_days)
nn.print_info(neural_network, X_test, y_test, False)

lr.create(X_train, y_train, test_name, save_as_file = True)
linear_regression = lr.load(test_name)
y_pred = lr.predict(linear_regression, data, X_test, pred_col, shown_days)
lr.print_info(linear_regression, X_test, y_test, y_pred)

bundesland = 'Niedersachsen'

kalenderwoche = 27
grossveranstaltung = 1 #ja
maskenpflicht = 1 #ja
kontaktbeschraenkung = 1 #ja
bundeslaender.predict_Data_for_onehot_encoded_bundesland(covid_data, kalenderwoche, pred_col, grossveranstaltung, maskenpflicht, kontaktbeschraenkung)
bundeslaender.predict_Data_for_one_bundesland(covid_data, kalenderwoche, pred_col, grossveranstaltung, bundesland,maskenpflicht, kontaktbeschraenkung)
bundeslaender.predict_data_with_knn_multi_label(covid_data, pred_col, kalenderwoche, grossveranstaltung,maskenpflicht,kontaktbeschraenkung)

