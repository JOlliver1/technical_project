import numpy as np
import matplotlib.pyplot as plt


def load_apple_data(data_path='/Users/James/Desktop/BristolUni/Year_4/TP/applemobilitytrends-2020-12-01.csv'):
    return np.loadtxt(data_path, delimiter=',', dtype=str, max_rows=4)


def load_google_data(data_path='/Users/James/Desktop/BristolUni/Year_4/TP/2020_GB_Region_Mobility_Report.csv'):
    return np.loadtxt(data_path, delimiter=',', dtype=str, max_rows=291)


# Apple Data

apple_data = load_apple_data()
apple_data = np.delete(apple_data, np.s_[125:127], axis=1)

driving = [float(i) for i in apple_data[1, 6:]]
transit = [float(i) for i in apple_data[2, 6:]]
walking = [float(i) for i in apple_data[3, 6:]]

plt.figure(figsize=(10, 5))
plt.plot(np.arange(12, 334), 100*np.ones(322), color='black', label='Baseline')
plt.plot(np.arange(12, 334), driving, label='Driving')
plt.plot(np.arange(12, 334), transit, label='Transit')
plt.plot(np.arange(12, 334), walking, label='Walking')
plt.ylabel('%')
plt.xlabel('Days')
plt.title('Apple Mobility Trends - UK')
plt.grid()
plt.legend()
plt.show()

