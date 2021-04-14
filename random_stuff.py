#import random_file
import math
import numpy as np
import matplotlib.pyplot as plt

x = np.zeros(20)

x[0] = 10
x[1] = 20
x[2] = 20
x[3] = 20
x[4] = 40

#x1 = np.count_nonzero(x < 30)
#print(x1)
y = ((5 < x) & (x < 30)).sum()
print(y)
