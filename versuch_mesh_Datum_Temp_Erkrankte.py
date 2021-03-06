#Nina Renken
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import pandas as pd
from datetime import date

data = pd.read_excel('Data\\Datum_Temp_Faelle_Mesh_0.xlsx')
data.Datum = data.Datum.str.replace("/", "")
data.Datum = data.Datum.astype(int)
#data = data.set_index('Datum')

fig = plt.figure()
ax = fig.gca(projection='3d')

X = data.Datum.to_numpy().astype(np.int64)
Y = np.arange(start=1, stop=18, step=1).astype(np.int64)
Z = data.loc[:, data.columns!='Datum'].to_numpy()

surf = ax.plot_surface(X, Y, Z)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

