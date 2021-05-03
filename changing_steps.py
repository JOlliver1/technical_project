from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import math
import datetime

from import_apple_data import average_uk

begin_time = datetime.datetime.now()


def find_dist(pos1, pos2):

    distance = math.sqrt(abs(pos1[0]-pos2[0])**2 + abs(pos1[1]-pos2[1])**2)

    return distance


def dist_check(pos1, pos2):

    distance = []

    for i in range(np.shape(pos2)[0]):
        distance.append(math.sqrt(abs(pos1[0]-pos2[i, 0])**2 + abs(pos1[1]-pos2[i, 1])**2))

    if all(x > 10 for x in distance):
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


def city_labeler(x):
    home_label = np.zeros(num)
    count = 0

    for i in range(np.shape(x)[1]):
        for j in range(np.shape(x)[0]):
            if x[j, i] != -1:
                home_label[count] = i+1
                count += 1

    return home_label


def make_colormap(seq):

    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}

    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])

    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


def colour_plotter(model):
    c = mcolors.ColorConverter().to_rgb
    rvb = make_colormap([c('black'), c('red'), 0.05, c('red'), c('yellow'), 0.5, c('yellow'), c('white')
                        , 0.9, c('white')])

    agent_counts = np.zeros((model.grid.width, model.grid.height))

    for cell in model.grid.coord_iter():
        cell_content, x, y = cell
        agent_count = len(cell_content)
        agent_counts[x][y] = agent_count

    plt.figure(figsize=(6, 6))
    plt.imshow(agent_counts.T, interpolation='nearest', cmap=rvb)
    plt.colorbar()
    plt.show()


def infected_plotter(model, day):

    no_infected = sum([1 for a in model.schedule.agents if a.infected == 1])
    infected_index_x = np.zeros(no_infected)
    index_x = np.zeros(model.num_agents)
    infected_index_y = np.zeros(no_infected)
    index_y = np.zeros(model.num_agents)
    count = 0
    count1 = 0

    for cell in model.grid.coord_iter():
        cell_content, x, y = cell
        for a in cell_content:
            index_x[count1] = x
            index_y[count1] = model.grid.height - y
            count1 += 1
            if a.infected == 1:
                infected_index_x[count] = x
                infected_index_y[count] = model.grid.height - y
                count += 1

    plt.rcParams['axes.facecolor'] = 'black'
    plt.figure(figsize=(6, 6))
    plt.scatter(index_x, index_y, marker='s', c='blue', s=0.5)
    plt.scatter(infected_index_x, infected_index_y, marker='s', c='red', s=0.5)
    plt.title('Step = ' + str(day) + '     No. Infected = ' + str(no_infected) + '/' + str(model.num_agents))
    plt.show()


class Agent(Agent):
    def __init__(self, unique_id, model, infection_rate, work_store, home_store, no_steps):
        super().__init__(unique_id, model)
        self.infected = 0
        self.working = 0
        self.infection = infection_rate
        self.work_store = work_store
        self.home_store = home_store
        self.no_steps = no_steps

    def spread_disease(self):
        if self.infected == 0:
            return

        else:
            cellmates = self.model.grid.get_cell_list_contents([self.pos])
            for a in cellmates:
                if a.infected != 1 and random.uniform(0, 1) < self.infection:
                    a.infected = 1

    def move(self):
        #day = math.floor(day_step/self.no_steps)
        if (day_step % self.no_steps) - 1 == 0:
            if self.work_store[self.unique_id, 0] != 0:
                new_position = (tuple(self.work_store[self.unique_id, :]))
                self.model.grid.move_agent(self, new_position)
                self.working = 1

        if (day_step % self.no_steps) - (self.no_steps-2) == 0:
            if self.work_store[self.unique_id, 0] != 0:
                new_position = (tuple(self.home_store[self.unique_id, :]))
                self.model.grid.move_agent(self, new_position)
                self.working = 0
        else:
            possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            while True:
                if self.working == 0:
                    new_position = self.random.choice(possible_steps)
                    if find_dist(new_position, self.home_store[self.unique_id, :]) <= 5:
                        self.model.grid.move_agent(self, new_position)
                        break
                elif self.working == 1:
                    new_position = self.random.choice(possible_steps)
                    if find_dist(new_position, self.work_store[self.unique_id, :]) <= 5:
                        self.model.grid.move_agent(self, new_position)
                        break

    def step(self):
        self.move()
        self.spread_disease()


def compute_informed(model):
    return sum([1 for a in model.schedule.agents if a.infected == 1])


def agent_locator(city_to_country, no_people, total_area, city_to_country_area, countryside, no_agents):
    num_agents = no_agents
    grid_size = round(math.sqrt((num_agents / no_people) * total_area) * 100)

    centers = np.zeros((1, 2))
    centers[0, :] = random.randrange(10, grid_size - 10), random.randrange(10, grid_size - 10)
    x = np.zeros((1, round(int(city_to_country * num_agents))))
    y = np.zeros((1, round(int(city_to_country * num_agents))))
    x[0, :] = np.around(np.random.normal(centers[0, 0], 3, round(int(city_to_country * num_agents))))
    y[0, :] = np.around(np.random.normal(centers[0, 1], 3, round(int(city_to_country * num_agents))))

    count = 0
    countryside_count = 0
    while countryside_count < (countryside * num_agents):
        countryside_count += counter(x)
        runner = True
        while runner:
            new_center = (random.randrange(10, grid_size - 10), random.randrange(10, grid_size - 10))
            if dist_check(new_center, centers):
                centers = np.vstack((centers, new_center))
                runner = False

        new_x = np.around(
            np.random.normal(centers[count, 0], (1 / (6 * city_to_country_area * (math.sqrt(count + 1))))
                             * grid_size, round(int(city_to_country * num_agents)
                                                      / (count + 2))))
        new_y = np.around(
            np.random.normal(centers[count, 1], (1 / (6 * city_to_country_area * (math.sqrt(count + 1))))
                             * grid_size, round(int(city_to_country * num_agents)
                                                       / (count + 2))))
        while len(new_x) < round(int(city_to_country * num_agents)):
            new_x = np.append(new_x, -1)
            new_y = np.append(new_y, -1)

        x = np.vstack((x, new_x))
        y = np.vstack((y, new_y))
        count += 1

    city_label = np.zeros(no_people)

    label = city_labeler(x)
    for i in range(len(label)):
        city_label[i] = label[i]

    new_x = np.delete(x.flatten(), np.where(x.flatten() == -1))
    new_y = np.delete(y.flatten(), np.where(y.flatten() == -1))

    x_countryside = np.around(np.random.uniform(0, grid_size - 1, int(num_agents - len(new_x))))
    y_countryside = np.around(np.random.uniform(0, grid_size - 1, int(num_agents - len(new_y))))

    all_x = np.concatenate((new_x, x_countryside))
    all_y = np.concatenate((new_y, y_countryside))

    return all_x, all_y, centers, city_label


class DiseaseModel(Model):
    def __init__(self, no_people, total_area, no_agents, Nc_N, n, all_x, all_y, centers, infection_rate, city_label,
                 no_steps):
        self.num_agents = no_agents
        grid_size = round(math.sqrt((self.num_agents / no_people) * total_area) * 100)
        self.grid = MultiGrid(grid_size, grid_size, False)
        self.schedule = RandomActivation(self)
        self.running = True

        flux_store = np.zeros((1, 3))
        home_store1 = np.zeros((self.num_agents, 2))

        for i in range(round(len(centers) / 2)):
            print(i, datetime.datetime.now() - begin_time)
            n_cities = random.sample(range(1, round(len(centers) / 2)), n)

            for j in range(len(n_cities)):
                mi = np.count_nonzero(city_label == i+1)
                nj = np.count_nonzero(city_label == n_cities[j])
                radius = math.sqrt((centers[i, 0] - centers[n_cities[j], 0]) ** 2 +
                                   (centers[i, 1] - centers[n_cities[j], 1]) ** 2)
                sij = 0

                for k in range(len(all_x)):
                    if (all_x[k] - centers[i, 0]) ** 2 + (all_y[k] - centers[i, 1]) ** 2 < radius ** 2:
                        sij += 1

                sij = sij - mi - nj
                if sij < 0:
                    sij = 0

                try:
                    Tij = (mi * Nc_N * mi * nj) / ((mi + sij) * (mi + nj + sij))*10
                except ZeroDivisionError:
                    Tij = 0

                if Tij > 75:
                    Tij = 75

                if Tij > 1 and (i != n_cities[j]):
                    flux_store = np.vstack((flux_store, (Tij, i+1, n_cities[j])))

        work_place = np.zeros(self.num_agents)
        work_store1 = np.zeros((num, 2))
        flux_store = np.delete(flux_store, 0, 0)

        for i in np.unique(flux_store[:, 1]):
            place = np.where(flux_store[:, 1] == i)[0]
            place1 = np.where(city_label == i)[0]
            for j in place1:
                for k in place:
                    if random.uniform(0, 100) < flux_store[k, 0]:
                        work_place[j] = flux_store[k, 2]

        for i in range(len(work_store1)):
            if work_place[i] != 0:
                n = int(work_place[i])
                work_store1[i, :] = centers[n, 0], centers[n, 1]

        for i in range(self.num_agents):
            home_store1[i, :] = int(all_x[i]), int(all_y[i])

        work_store = np.int64(work_store1)
        home_store = np.int64(home_store1)

        for i in range(self.num_agents):
            a = Agent(i, self, infection_rate, work_store, home_store, no_steps)
            self.schedule.add(a)
            self.grid.place_agent(a, (int(all_x[i]), int(all_y[i])))

            if i == 1:
                a.infected = 1

        self.datacollector = DataCollector(
            model_reporters={"Tot informed": compute_informed},
            agent_reporters={"Infected": "infected"})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


num = 2000

city_to_country = 0.14
no_people = 67000000
total_area = 240000
city_to_country_area = 13
countryside = 0.8

infection_rate = 0.1
n = 50

all_x, all_y, centers, city_label = agent_locator(city_to_country,
                                      no_people,
                                      total_area,
                                      city_to_country_area,
                                      countryside,
                                      num)

model = DiseaseModel(no_people=67000000,
                     total_area=240000,
                     no_agents=num,
                     Nc_N=0.2,
                     n=n,
                     all_x=all_x,
                     all_y=all_y,
                     centers=centers,
                     infection_rate=infection_rate,
                     city_label=city_label,
                     no_steps=5)

print('model init')

steps = 50*5
for day_step in range(steps):
    model.step()
    print(day_step, datetime.datetime.now() - begin_time)

#############################################################################

model1 = DiseaseModel(no_people=67000000,
                      total_area=240000,
                      no_agents=num,
                      Nc_N=0.2,
                      n=n,
                      all_x=all_x,
                      all_y=all_y,
                      centers=centers,
                      infection_rate=infection_rate,
                      city_label=city_label,
                      no_steps=8)

print('model1 init')

steps1 = 50*8
for day_step in range(steps1):
    model1.step()
    print(day_step, datetime.datetime.now() - begin_time)

#############################################################################

model2 = DiseaseModel(no_people=67000000,
                      total_area=240000,
                      no_agents=num,
                      Nc_N=0.2,
                      n=n,
                      all_x=all_x,
                      all_y=all_y,
                      centers=centers,
                      infection_rate=infection_rate,
                      city_label=city_label,
                      no_steps=12)

print('model2 init')

steps2 = 50*12
for day_step in range(steps2):
    model2.step()
    print(day_step, datetime.datetime.now() - begin_time)

#############################################################################

model3 = DiseaseModel(no_people=67000000,
                      total_area=240000,
                      no_agents=num,
                      Nc_N=0.2,
                      n=n,
                      all_x=all_x,
                      all_y=all_y,
                      centers=centers,
                      infection_rate=infection_rate,
                      city_label=city_label,
                      no_steps=20)

print('model3 init')

steps3 = 50*20
for day_step in range(steps3):
    model3.step()
    print(day_step, datetime.datetime.now() - begin_time)

#############################################################################

model4 = DiseaseModel(no_people=67000000,
                      total_area=240000,
                      no_agents=num,
                      Nc_N=0.2,
                      n=n,
                      all_x=all_x,
                      all_y=all_y,
                      centers=centers,
                      infection_rate=infection_rate,
                      city_label=city_label,
                      no_steps=30)

print('model4 init')

steps4 = 50*30
for day_step in range(steps4):
    model4.step()
    print(day_step, datetime.datetime.now() - begin_time)

#############################################################################

infection_panda = model.datacollector.get_agent_vars_dataframe()
infection_np = infection_panda.to_numpy()

infection_matrix = np.reshape(infection_np, (num, 50*5), order='F')
infection_array = np.zeros(len(range(0, (50*5)-5, 5)))

for i in range(0, (50*5)-5, 5):
    count = 0
    for j in range(2000):
        if infection_matrix[j, i] == 1:
            count += 1

    infection_array[int(i / 5)] = count

#############################################################################

infection_panda1 = model1.datacollector.get_agent_vars_dataframe()
infection_np1 = infection_panda1.to_numpy()

infection_matrix1 = np.reshape(infection_np1, (num, 50*8), order='F')
infection_array1 = np.zeros(len(range(0, (50*8)-8, 8)))

for i in range(0, (50*8)-8, 8):
    count = 0
    for j in range(2000):
        if infection_matrix1[j, i] == 1:
            count += 1

    infection_array1[int(i / 8)] = count

#############################################################################

infection_panda2 = model2.datacollector.get_agent_vars_dataframe()
infection_np2 = infection_panda2.to_numpy()

infection_matrix2 = np.reshape(infection_np2, (num, 50*12), order='F')
infection_array2 = np.zeros(len(range(0, (50*12)-12, 12)))

for i in range(0, (50*12)-12, 12):
    count = 0
    for j in range(2000):
        if infection_matrix2[j, i] == 1:
            count += 1

    infection_array2[int(i / 12)] = count

#############################################################################

infection_panda3 = model3.datacollector.get_agent_vars_dataframe()
infection_np3 = infection_panda3.to_numpy()

infection_matrix3 = np.reshape(infection_np3, (num, 50*20), order='F')
infection_array3 = np.zeros(len(range(0, (50*20)-20, 20)))

for i in range(0, (50*20)-20, 20):
    count = 0
    for j in range(2000):
        if infection_matrix3[j, i] == 1:
            count += 1

    infection_array3[int(i / 20)] = count

#############################################################################

infection_panda4 = model4.datacollector.get_agent_vars_dataframe()
infection_np4 = infection_panda4.to_numpy()

infection_matrix4 = np.reshape(infection_np4, (num, 50*30), order='F')
infection_array4 = np.zeros(len(range(0, (50*30)-30, 30)))

for i in range(0, (50*30)-30, 30):
    count = 0
    for j in range(2000):
        if infection_matrix4[j, i] == 1:
            count += 1

    infection_array4[int(i / 30)] = count

#############################################################################

plt.rcParams['axes.facecolor'] = 'white'
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

ax.plot(np.arange(0, 49), infection_array, color='blue', label='steps=5')
ax.plot(np.arange(0, 49), infection_array1, color='red', label='steps=8')
ax.plot(np.arange(0, 49), infection_array2, color='green', label='steps=12')
ax.plot(np.arange(0, 49), infection_array3, color='orange', label='steps=20')
ax.plot(np.arange(0, 49), infection_array4, color='pink', label='steps=30')
ax.set_xlabel('Days')
ax.set_ylabel('No. of People Infected')
ax.legend()
ax.grid(linestyle='--')

plt.tight_layout()
plt.show()  # """

#############################################################################

out = model.datacollector.get_agent_vars_dataframe().groupby('Step').sum()
new_out = out.to_numpy()

#############################################################################

out1 = model1.datacollector.get_agent_vars_dataframe().groupby('Step').sum()
new_out1 = out1.to_numpy()

#############################################################################

out2 = model2.datacollector.get_agent_vars_dataframe().groupby('Step').sum()
new_out2 = out2.to_numpy()

#############################################################################

out3 = model3.datacollector.get_agent_vars_dataframe().groupby('Step').sum()
new_out3 = out3.to_numpy()

#############################################################################

out4 = model4.datacollector.get_agent_vars_dataframe().groupby('Step').sum()
new_out4 = out4.to_numpy()

#############################################################################

fig1, ax1 = plt.subplots(1, 1, figsize=(10, 5))

ax1.plot(np.arange(0, len(new_out)), new_out, color='blue', label='steps=5')
ax1.plot(np.arange(0, len(new_out1)), new_out1, color='red', label='isteps=8')
ax1.plot(np.arange(0, len(new_out2)), new_out2, color='green', label='steps=12')
ax1.plot(np.arange(0, len(new_out3)), new_out3, color='orange', label='steps=20')
ax1.plot(np.arange(0, len(new_out4)), new_out4, color='pink', label='steps=30')
ax1.set_xlabel('Steps')
ax1.set_ylabel('No. People Infected')
ax1.legend()
ax1.grid(linestyle='--')
#secax = ax.secondary_xaxis(-0.15, functions=(lambda x: x/8, lambda x: x/8))
#secax.set_xlabel('Days')

plt.tight_layout()
plt.show()  # """

print(datetime.datetime.now() - begin_time)

