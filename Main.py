import DataReader as reader
import Heatmap as hm
import Charts as ch
import LinReg as lr
import NeuralNetwork as nn
import Util as u

covid_data = reader.get_covid_data()

#ch.generate_pie_chart(covid_data)
#ch.generate_bar_chart(covid_data)
#ch.generate_graph(covid_data)

#hm.generate_chart(covid_data)
#hm.generate_circle(covid_data)
#hm.generate_heatmap(covid_data)

test_name = 'P_Test1'
pred_col = 'AnzahlFall'
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