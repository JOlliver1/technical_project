import numpy as np
import matplotlib.pyplot as plt


def load_google_data(data_path='/Users/James/Desktop/BristolUni/Year_4/TP/2020_GB_Region_Mobility_Report.csv'):
    return np.loadtxt(data_path, delimiter=',', dtype=str, max_rows=290)


# Google Data

google_data = load_google_data()
google_data = np.delete(google_data, np.s_[:8], axis=1)
retail = [float(i) for i in google_data[1:, 0]]
grocery = [float(i) for i in google_data[1:, 1]]
parks = [float(i) for i in google_data[1:, 2]]
transport = [float(i) for i in google_data[1:, 3]]
workplaces = [float(i) for i in google_data[1:, 4]]
resident = [float(i) for i in google_data[1:, 5]]

plt.figure(figsize=(10, 5))
plt.plot(np.arange(15, 304), resident)
plt.grid()
plt.show()
