from sklearn import decomposition
from AQI_DataSet import *
import pandas as pd
import matplotlib.pyplot as plt

x_train, x_test, y_train, y_test = DataSet_Classify_Random(3)
name = ['PM2.5', 'PM10', 'SO2', 'CO', 'NO2', 'O3_8h']

fa = decomposition.FactorAnalysis(n_components=2)
X = fa.fit_transform(x_train)

pos = pd.DataFrame(X, columns=['X', 'Y'])
print(pos)
pos['name'] = y_train

ax = pos[pos['name'] == 0].plot(kind='scatter', x='X', y='Y', color='blue', label=0)
pos[pos['name'] == 1].plot(kind='scatter', x='X', y='Y', color='green', label=1, ax=ax)
pos[pos['name'] == 2].plot(kind='scatter', x='X', y='Y', color='red', label=2, ax=ax)
pos[pos['name'] == 3].plot(kind='scatter', x='X', y='Y', color='yellow', label=3, ax=ax)
plt.show()
