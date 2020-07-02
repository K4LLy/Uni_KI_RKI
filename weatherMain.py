#Pascal und Nina
import CovidWeather as cw
import DataReader as reader

from IPython import get_ipython

bundesland = 'Niedersachsen'

get_ipython().run_line_magic('matplotlib', 'qt')

covid_data = reader.get_covid_data()
cw.get_covid_weather(bundesland, covid_data)