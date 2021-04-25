#import random_file
import math
import numpy as np
import matplotlib.pyplot as plt

a= np.array((1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

plt.rcParams['axes.facecolor'] = 'white'
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.plot(np.arange(0, 10), a, color='blue', label='Infection=1.0')
ax.set_xlabel('Days')
ax.set_ylabel('No. of People Infected')
ax.legend()
ax.grid(linestyle='--')
secax = ax.secondary_xaxis(-0.15, functions=(lambda x: 8*x, lambda x: 8*x))
secax.set_xlabel('Steps')

plt.tight_layout()
plt.show()  # """

