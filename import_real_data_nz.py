import numpy as np
import matplotlib.pyplot as plt
import random

from import_apple_data import *


def load_real_data(data_path='/Users/James/Downloads/owid-covid-data.csv'):
    return np.loadtxt(data_path, delimiter=',', dtype=str, skiprows=1, usecols=5), \
           np.loadtxt(data_path, delimiter=',', dtype=str, skiprows=1, usecols=3)

# Real Data


data_nz, dates = load_real_data()
data_nz = data_nz[:-1][:142]
dates = dates[:-1][:142]

for i in range(len(data_nz)):
    if int(data_nz[i]) < 0:
        data_nz[i] = 0
    else:
        data_nz[i] = int(data_nz[i])

data_nz = data_nz.astype(np.int)

"""
data_nz = data_nz[::-1]#[31:173]
dates = dates[::-1]#[31:173]
#average_uk = average_uk[38:181]"""

fig, ax = plt.subplots(2, 1, figsize=(10, 6))

#par = ax[0].twinx()

ax[1].bar(np.arange(0, len(data_nz)), data_nz, color='red', width=1.0)
ax[1].set_ylabel('Daily Cases')
ax[1].set_xlabel('Days')
#ax[0].axes.get_xaxis().set_visible(False)
ax[0].plot(np.arange(0, len(data_nz)), np.cumsum(data_nz), color='red')
#par.plot(np.arange(0, len(average_uk)), average_uk, color='blue')
ax[0].set_ylabel('Total Cases')
ax[0].grid(linestyle='--')
plt.show()

