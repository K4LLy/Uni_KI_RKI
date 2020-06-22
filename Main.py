import DataReader as reader
import Heatmap as hm
import Charts as ch
import LinReg as lr

data = reader.get_covid_data()

#ch.generate_pie_chart(data)
#ch.generate_bar_chart(data)
#ch.generate_graph(data)

#hm.generate_chart(data)
#hm.generate_circle(data)
#hm.generate_heatmap(data)

#lr.predict(data, 15, 70, 80, 'AnzahlGenesen')
lr.predict(data, 15, 40, 80, 'AnzahlFall', True)