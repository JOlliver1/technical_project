import numpy as np
import matplotlib.pyplot as plt
import random


def load_apple_data(data_path='/Users/James/Desktop/BristolUni/Year_4/TP/applemobilitytrends-2020-12-01.csv'):
    return np.loadtxt(data_path, delimiter=',', dtype=str, max_rows=4)


# Apple Data

apple_data = load_apple_data()
apple_data = np.delete(apple_data, np.s_[125:127], axis=1)

driving = [float(i) for i in apple_data[1, 6:]]
transit = [float(i) for i in apple_data[2, 6:]]
walking = [float(i) for i in apple_data[3, 6:]]

average = np.mean([driving, walking, transit], axis=0)[33:]
new_average = np.concatenate((average[:28], average[:28], average[:-56]), axis=None)
newer_average = np.concatenate((average[28:], average[-28:]), axis=None)

"""
plt.figure(figsize=(10, 5))
plt.plot(np.arange(0, 289), 100*np.ones(289), color='black', label='Baseline')
#plt.plot(np.arange(12, 334), driving, label='Driving')
#plt.plot(np.arange(12, 334), transit, label='Transit')
#plt.plot(np.arange(12, 334), walking, label='Walking')
plt.plot(np.arange(0, 289), new_average, label='Behind', color='red')
plt.plot(np.arange(0, 289), newer_average, label='Ahead', color='green')
plt.plot(np.arange(0, 289), average, label='Real', color='blue')
plt.ylabel('%')
plt.xlabel('Days')
plt.grid()
plt.legend()
plt.show()  # """

#plt.figure(figsize=(10, 5))
#plt.plot(np.arange(12, 334), np.mean([driving, walking, transit], axis=0))
#plt.show()

