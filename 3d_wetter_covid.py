#Nina Renken
import numpy as np
import matplotlib.pyplot as plt
import DataReader as reader
import datetime as dt
import pandas as pd


data_weather = reader.get_weather_data()
data_weather_avg = data_weather.groupby('Zeitstempel').mean()
data_covid = reader.get_covid_data()
data_covid_nds = data_covid[data_covid.Bundesland.eq('Niedersachsen')]
data_covid_avg = data_covid_nds.groupby('Meldedatum').sum()
data_covid_avg = data_covid_avg.reset_index()

# Präparieren der Wetterdaten
df_list = []
for row in data_weather_avg.itertuples():
    datum = row.Index
    zeitstempel = dt.datetime.strptime(str(datum), '%Y%m%d').strftime('%Y/%m/%d')
    df_list.append([zeitstempel, row.Wert])

df_weather = pd.DataFrame(df_list, columns=['Datum', 'Temperatur'])
df_weather.plot(x='Datum', y='Temperatur', rot=90)

data_covid_avg.plot(x='Meldedatum', y='AnzahlFall', rot=90)

data = pd.merge(data_covid_avg, df_weather, left_on='Meldedatum', right_on='Datum')




def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zlow, zhigh)
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()




"""
x = data.Datum
y = data.AnzahlFall
z = data.Temperatur

z = np.expand_dims(z, axis = 0)
z = np.repeat(z, repeats = len(x), axis = 0)

x = np.expand_dims(x, axis = 1)
x = np.repeat(x, repeats = len(x), axis = 1)

y = np.expand_dims(y, axis = 0)
y = np.repeat(y, repeats = len(x), axis = 0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

Axes3D.plot_surface(ax, data.Datum, data.AnzahlFall, data.Temperatur)
"""
#Z_list = []
#Z = Z_list.append([data.Temperatur, data.Datum])
#surf = ax.plot_surface(data.Datum, data.AnzahlFall, data.Temperatur, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#surf = ax.contour3D(data.Datum, data.AnzahlFall, data.Temperatur, cmap='binary')

#ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_formatter(FormatStrFormatter('%.02f'))

#fig.colorbar(surf, shrink=0.5, aspect=5)

#plt.show()



"""
x, y = np.mgrid[-3:3:30j, -3_3_30j]
z = (x**2+y**3)*np.exp(-x**2-y**2)
cmap = 'coolwarm'

fig = plt.figure()
ax = fig.csa(projection='3d')
ax.pllot_surface(x, y, z, rstride=1, cstride=1, cmap=cmap, alpha=0,5)
cset = ax.contourf(x, y, z, zdir='z', offset=-0,8, cmap=cmap)
ax.set_xlabel('$x$', size='xx-large')
ax.set_ylabel('$y$', size='xx-large')
ax.set_zlabel('$z$', size='xx-large')
ax.set_zlim(-0.8, 0.5)

plt.draw()
"""