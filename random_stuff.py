from import_apple_data import *
import math
import numpy as np

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))

par1 = ax.twinx()

ax.set_xlabel("Steps")
ax.set_ylabel("No. People Infected")
par1.set_ylabel("Mobility %")

p1, = ax.plot(np.arange(0, len(spread_average_uk)), np.random.normal(size=len(spread_average_uk)), color='red', label="No Mobility Data")
p2, = ax.plot(np.arange(0, len(spread_average_uk)), np.random.normal(size=len(spread_average_uk)), color='green', label="Mobility Data")
p3, = par1.plot(np.arange(0, len(spread_average_uk)), spread_average_uk, color='blue', label="Mobility %")

ax.legend(handles=[p1, p2], loc='best')

par1.yaxis.label.set_color('blue')
ax.grid(linestyle='--')

secax = ax.secondary_xaxis(-0.15, functions=(lambda x: x/8, lambda x: x/8))
secax.set_xlabel('Days')

fig.tight_layout()
plt.show()

