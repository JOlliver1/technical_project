import random_file
import math
import numpy as np
import matplotlib.pyplot as plt

population = 2000
city_ratio = 0.83  # for UK
city_to_country = 0.14  # for UK
grid_width = 200
grid_height = 200


def find_dist(pos1, pos2):

    distance = math.sqrt(abs(pos1[0]-pos2[0])**2 + abs(pos1[1]-pos2[1])**2)

    return distance


center1 = np.array((random_file.randrange(20, 180), random_file.randrange(20, 180)))
# center2 = (random.randrange(40, 160), random.randrange(40, 160))
# center3 = (random.randrange(40, 160), random.randrange(40, 160))
# center4 = (random.randrange(40, 160), random.randrange(40, 160))

while True:
    center2 = (random_file.randrange(20, 180), random_file.randrange(20, 180))
    if find_dist(center1, center2) > 30:
        break

while True:
    center3 = (random_file.randrange(20, 180), random_file.randrange(20, 180))
    if find_dist(center1, center3) > 30 and find_dist(center2, center3) > 30:
        break

while True:
    center4 = (random_file.randrange(20, 180), random_file.randrange(20, 180))
    if find_dist(center1, center4) > 30 and find_dist(center2, center4) > 30 and find_dist(center3, center4) > 30:
        break

x1 = np.around(np.random.normal(center1[0], 5, 500))
y1 = np.around(np.random.normal(center1[1], 5, 500))

x2 = np.around(np.random.normal(center2[0], 5, 250))
y2 = np.around(np.random.normal(center2[1], 5, 250))

x3 = np.around(np.random.normal(center3[0], 5, 125))
y3 = np.around(np.random.normal(center3[1], 5, 125))

x4 = np.around(np.random.normal(center4[0], 5, 62))
y4 = np.around(np.random.normal(center4[1], 5, 62))

x5 = np.around(np.random.uniform(0, grid_width, int((1-city_ratio)*2000)))
y5 = np.around(np.random.uniform(0, grid_height, int((1-city_ratio)*2000)))

all_x = np.concatenate((x1, x2, x3, x4, x5))
all_y = np.concatenate((y1, y2, y3, y4, y5))
all = np.vstack((all_x, all_y))
agent_counts = np.zeros((grid_height+1, grid_width+1))

for i in range(np.shape(all)[1]):
    if 0 < int(all[0, i]) & int(all[1, i]) < 200:
        agent_counts[int(all[0, i]), 200-int(all[1, i])] += 1

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(x5, y5)
plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.scatter(x3, y3)
plt.scatter(x4, y4)

plt.subplot(1, 2, 2)
plt.imshow(agent_counts.T, interpolation='nearest', cmap='hot')
plt.colorbar()
plt.show()
