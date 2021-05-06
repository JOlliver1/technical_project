import numpy as np
import matplotlib.pyplot as plt
import random

from import_apple_data import *


def load_real_data(data_path='/Users/James/Downloads/data_2021-Apr-29.csv'):
    return np.loadtxt(data_path, delimiter=',', dtype=int, skiprows=19, usecols=4), \
           np.loadtxt(data_path, delimiter=',', dtype=str, skiprows=19, usecols=3)


# Real Data

data_uk,  dates = load_real_data()

#print(dates)

data_uk = data_uk[::-1][31:173]
dates = dates[::-1][31:173]
average_uk = average_uk[38:181]

data_uk_cum = np.cumsum(data_uk)

spread_data_uk = np.zeros(len(data_uk_cum)*8)

for i in range(len(data_uk_cum)):
    for j in range(0, 8):
        spread_data_uk[(i*8)+j] = data_uk_cum[i]

"""fig, ax = plt.subplots(2, 1, figsize=(10, 6))

par = ax[0].twinx()

ax[1].bar(np.arange(0, len(data_uk)), data_uk, color='red', width=1.0)
ax[1].set_ylabel('Daily Cases')
#ax[0].axes.get_xaxis().set_visible(False)
ax[0].plot(np.arange(0, len(data_uk)), np.cumsum(data_uk), color='red')
par.plot(np.arange(0, len(average_uk)), average_uk, color='blue')
ax[0].set_ylabel('Total Cases')
ax[1].set_xlabel('Days')
ax[0].grid(linestyle='--')
plt.show()"""

"""plt.figure(figsize=(10, 5))
plt.plot(np.arange(0, len(spread_average_uk)), 100*np.ones(len(spread_average_uk)), color='black', label='Baseline')
plt.plot(np.arange(0, len(spread_average_uk)), spread_average_uk, label='Average', color='red')
plt.ylabel('%')
plt.xlabel('Days')
plt.grid(linestyle='--')
plt.legend()
plt.show()"""


"""
new_average = np.concatenate((average[:28], average[:28], average[:-56]), axis=None)
newer_average = np.concatenate((average[28:], average[-28:]), axis=None)"""

"""
plt.figure(figsize=(10, 5))
plt.plot(np.arange(0, len(driving_uk)), 100*np.ones(len(driving_uk)), color='black', label='Baseline')
plt.plot(np.arange(0, len(driving_uk)), driving_uk, label='Driving', color='red')
plt.plot(np.arange(0, len(driving_uk)), transit_uk, label='Transit', color='green')
plt.plot(np.arange(0, len(driving_uk)), walking_uk, label='Walking', color='blue')
plt.plot(np.arange(0, len(driving_uk)), average_uk, label='Average', color='orange')
plt.ylabel('%')
plt.xlabel('Days')
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(np.arange(0, len(driving_nz)), 100*np.ones(len(driving_nz)), color='black', label='Baseline')
plt.plot(np.arange(0, len(driving_nz)), driving_nz, label='Driving', color='red')
plt.plot(np.arange(0, len(driving_nz)), transit_nz, label='Transit', color='green')
plt.plot(np.arange(0, len(driving_nz)), walking_nz, label='Walking', color='blue')
plt.plot(np.arange(0, len(driving_nz)), average_nz, label='Average', color='orange')
plt.ylabel('%')
plt.xlabel('Days')
plt.grid()
plt.legend()
plt.show()  # """


