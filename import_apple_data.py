import numpy as np
import matplotlib.pyplot as plt
import random


def load_apple_data(data_path='/Users/James/Desktop/BristolUni/Year_4/TP/applemobilitytrends-2020-12-01.csv'):
    return np.loadtxt(data_path, delimiter=',', dtype=str, max_rows=4)


def load_apple_data2(data_path='/Users/James/Desktop/BristolUni/Year_4/TP/applemobilitytrends-2021-04-20.csv'):
    return np.loadtxt(data_path, delimiter=',', dtype=str, skiprows=144, max_rows=3), \
           np.loadtxt(data_path, delimiter=',', dtype=str, max_rows=1), \
           np.loadtxt(data_path, delimiter=',', dtype=str, skiprows=91, max_rows=3)


# Apple Data

apple_data_uk, dates, apple_data_nz = load_apple_data2()
apple_data_uk = np.delete(apple_data_uk, np.s_[125:127], axis=1)
apple_data_uk = np.delete(apple_data_uk, 428, axis=1)
apple_data_nz = np.delete(apple_data_nz, np.s_[125:127], axis=1)
apple_data_nz = np.delete(apple_data_nz, 428, axis=1)
dates = np.delete(dates, np.s_[125:127], axis=None)
dates = np.delete(dates, 428, axis=None)

driving_uk = [float(i) for i in apple_data_uk[0, 6:]]
transit_uk = [float(i) for i in apple_data_uk[1, 6:]]
walking_uk = [float(i) for i in apple_data_uk[2, 6:]]
driving_nz = [float(i) for i in apple_data_nz[0, 6:]]
transit_nz = [float(i) for i in apple_data_nz[1, 6:]]
walking_nz = [float(i) for i in apple_data_nz[2, 6:]]

average_uk = np.mean([driving_uk, walking_uk, transit_uk], axis=0)
average_nz = np.mean([driving_nz, walking_nz, transit_nz], axis=0)

average_uk = average_uk[:-11]
average_nz = average_nz[:-11]

#print(len(average_uk))
spread_average_uk = np.zeros(len(average_uk)*8)
spread_average = spread_average_uk
spread_average_nz = np.zeros(len(average_nz)*8)

for i in range(len(average_uk)):
    for j in range(0, 8):
        spread_average_uk[(i*8)+j] = average_uk[i]
        spread_average_nz[(i*8)+j] = average_nz[i]

spread_average_uk = spread_average_uk[304:1448]#:1448]  # 891
spread_average_nz = spread_average_nz[304:1448]  #
spread_average_uk2 = spread_average[192:1336]  #
spread_average_uk1 = spread_average[416:1560]  # later shift

split_average_uk1 = spread_average_uk[:208]
split_average_uk2 = spread_average_uk[208:]

spreaduk2 = np.zeros(len(spread_average_uk2))
spreaduk1 = np.zeros(len(spread_average_uk1))
spreaduk = np.zeros(len(spread_average_uk))

for i in range(len(spreaduk2)):
    if i > 422:
        spreaduk2[i] = 20
    else:
        spreaduk2[i] = spread_average_uk2[i]

for i in range(len(spreaduk1)):
    if i > 198:
        spreaduk1[i] = 20
    else:
        spreaduk1[i] = spread_average_uk1[i]

for i in range(len(spreaduk)):
    if i > 310:
        spreaduk[i] = 20
    else:
        spreaduk[i] = spread_average_uk[i]

# plt.plot(0, len(spread_average_uk), spreaduk, color='b')
# plt.plot(0, len(spread_average_uk1), spreaduk1, color='r')
# plt.plot(0, len(spread_average_uk2), spreaduk2, color='g')
# plt.show()

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


