import DataReader as reader
import Heatmap as hm
import Charts as ch
import Heatmap_chart_test as hmt
#import HeatmapRichtig as hmr

data = reader.getCovidData()


#ch.generate_pie_chart(data)
#ch.generate_bar_chart(data)
ch.generate_graph(data)

#hm.generate(data, "Map")

