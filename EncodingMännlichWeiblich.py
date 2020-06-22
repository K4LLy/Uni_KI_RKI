# -*- coding: utf-8 -*-

import DataReader as reader
import numpy as np
covid_data = reader.get_covid_data()
#geschlechter = np.array(covid_data.filter(items=['Geschlecht']))

covid_data = covid_data.groupby(['Meldedatum', 'Geschlecht']).sum()
geschlechter = np.array(covid_data.index.get_level_values(1))
covid_data.loc[:,('Geschlecht2')] = geschlechter
#covid_data.reset_index(level=0, inplace=True)
values = np.array(covid_data.filter(items=['Geschlecht2']))
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# define example


# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded) #0 = männlich, 1 = weiblich, 2 = unbekannt
covid_data.loc[:,('maennlich')] = onehot_encoded[ :, 0]
covid_data.loc[:,('weiblich')] = onehot_encoded[ :, 1]
covid_data.loc[:,('unbekannt')] = onehot_encoded[ :, 2]
covid_data=covid_data.drop(['Geschlecht2'], 1)
# invert first example
#inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
#print(inverted)