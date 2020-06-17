import DataReader as reader
import Heatmap as hm
import Charts as ch

data = reader.get_covid_data()


#ch.generate_pie_chart(data)
#ch.generate_bar_chart(data)
ch.generate_graph(data)

#hm.generate(data, "Map")

