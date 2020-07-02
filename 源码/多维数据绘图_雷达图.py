import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from AQI_DataSet import *

x_train, x_test, y_train, y_test = DataSet_Classify_Random(3)
name = ['PM2.5', 'PM10', 'SO2', 'CO', 'NO2', 'O3_8h']
data_dict = {}
for i in range(len(name)):
    data_dict[name[i]] = np.array(x_train[:, i])
data_dict["label"] = y_train
# print(data_dict)
# 合成dataFrame
pd_data = pd.DataFrame(data_dict)
print(pd_data)
# 画图
plt.figure()
pd.plotting.radviz(pd_data, "label", color=['red', 'blue', 'yellow', 'green'])
plt.show()
