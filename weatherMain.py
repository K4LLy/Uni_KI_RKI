#Pascal und Nina
import CovidWeather as cw
import DataReader as reader
import Heatmap as hm
import Funkmasten as fm

from IPython import get_ipython

bundesland = 'Niedersachsen'

get_ipython().run_line_magic('matplotlib', 'qt')

covid_data = reader.get_covid_data()

hm.generate_circle(covid_data)

hm.generate_heatmap(covid_data)

fm.generate_heatmap_5g(covid_data, 'Heatmap_5G')

cw.get_covid_weather(bundesland, covid_data)