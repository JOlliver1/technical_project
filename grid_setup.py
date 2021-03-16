import random
import math
import numpy as np
import matplotlib.pyplot as plt

population = 2000
city_to_country = 0.14  # for UK
grid_width = 200
grid_height = 200


def dist_check(pos1, pos2):

    distance = []

    for i in range(np.shape(pos2)[0]):
        distance.append(math.sqrt(abs(pos1[0]-pos2[i, 0])**2 + abs(pos1[1]-pos2[i, 1])**2))

    if all(x > 25 for x in distance):
        return True
    else:
        return False


def counter(array):

    count = 0

    if np.shape(array)[0] < 2:
        for j in range(np.shape(array)[1]):
            if j != -1:
                count += 1
    else:
        for j in range(np.shape(array)[1]):
            if array[-1, j] != -1:
                count += 1

    return count


def counter1(array):

    count = 0

    for i in range(np.shape(array)[0]):
        for j in range(len(array[i, :])):
            if array[i, j] != -1:
                count += 1

    return count


centers = np.zeros((1, 2))
centers[0, :] = random.randrange(10, grid_width-10), random.randrange(10, grid_height-10)
x = np.zeros((1, round(int(city_to_country*population))))
y = np.zeros((1, round(int(city_to_country*population))))
x[0, :] = np.around(np.random.normal(centers[0, 0], 3, round(int(city_to_country*population))))
y[0, :] = np.around(np.random.normal(centers[0, 1], 3, round(int(city_to_country*population))))
#print(x[:, ])

count = 0
counter(x)

while counter(x) > 1:
    runner = True
    while runner:
        new_center = (random.randrange(10, grid_width-10), random.randrange(10, grid_height-10))
        if dist_check(new_center, centers) & count < 20:
            centers = np.vstack((centers, new_center))
            runner = False

    new_x = np.around(np.random.normal(centers[count, 0], 3, round(int(city_to_country*population)/(count+2))))
    new_y = np.around(np.random.normal(centers[count, 1], 3, round(int(city_to_country*population)/(count+2))))
    while len(new_x) < round(int(city_to_country*population)):
        new_x = np.append(new_x, -1)
        new_y = np.append(new_y, -1)

    x = np.vstack((x, new_x))
    y = np.vstack((y, new_y))
    count += 1


new_x = np.delete(x.flatten(), np.where(x.flatten() == -1))
new_y = np.delete(y.flatten(), np.where(y.flatten() == -1))

x5 = np.around(np.random.uniform(0, grid_width, int(population-len(new_x))))
y5 = np.around(np.random.uniform(0, grid_height, int(population-len(new_y))))

all_x = np.concatenate((new_x, x5))
all_y = np.concatenate((new_y, y5))

all_xy = np.vstack((all_x, all_y))
agent_counts = np.zeros((grid_height+1, grid_width+1))

for i in range(np.shape(all_xy)[1]):
    if 0 < int(all_xy[0, i]) & int(all_xy[1, i]) < 200:
        agent_counts[int(all_xy[0, i]), 200-int(all_xy[1, i])] += 1

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(all_x, all_y)
#plt.scatter(x5, y5)
#plt.scatter(x, y)

plt.subplot(1, 2, 2)
plt.imshow(agent_counts.T, interpolation='nearest', cmap='hot')
plt.colorbar()
plt.show()
