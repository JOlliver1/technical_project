from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.batchrunner import BatchRunner

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import math
import datetime

from import_apple_data import *

spread1 = spread_average_uk
spread2 = spread_average_uk1
spread3 = spread_average_uk2

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
    def __init__(self, unique_id, model, infection_rate, work_store, home_store, mobility, stepper):
        super().__init__(unique_id, model)
        self.infected = 0
        self.working = 0
        self.rnumber = 0
        self.infection = infection_rate
        self.work_store = work_store
        self.home_store = home_store
        self.mobility = mobility
        self.stepper = stepper

    def spread_disease(self):
        if self.infected == 0:
            return

        else:
            cellmates = self.model.grid.get_cell_list_contents([self.pos])
            for a in cellmates:
                if a.infected != 1 and random.uniform(0, 1) < self.infection:
                    a.infected = 1
                    self.rnumber += 1

    def move(self):
        if random.uniform(0, 1) < self.mobility:
            if (self.stepper % 8) - 2 == 0:
                if self.work_store[self.unique_id, 0] != 0:
                    new_position = (tuple(self.work_store[self.unique_id, :]))
                    self.model.grid.move_agent(self, new_position)
                    self.working = 1

            elif (self.stepper % 8) - 6 == 0:
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


def rnumber_calc(model):
    count = 0
    sum_rnumber = 0
    for a in model.schedule.agents:
        if a.infected == 1:
            count += 1
            sum_rnumber += a.rnumber

    return sum_rnumber/count


def agent_locator(city_to_country, no_people, total_area, city_to_country_area, countryside, no_agents, n, Nc_N):
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

    all_x[1] = centers[0, 0]
    all_y[1] = centers[0, 1]

    flux_store = np.zeros((1, 3))
    home_store1 = np.zeros((num_agents, 2))

    for i in range(round(len(centers) / 2)):
        print(i, datetime.datetime.now() - begin_time)
        n_cities = random.sample(range(1, round(len(centers) / 2)), n)

        for j in range(len(n_cities)):
            mi = np.count_nonzero(city_label == i + 1)
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
                Tij = (mi * Nc_N * mi * nj) / ((mi + sij) * (mi + nj + sij)) * 10
            except ZeroDivisionError:
                Tij = 0

            if Tij > 75:
                Tij = 75

            if Tij > 1 and (i != n_cities[j]):
                flux_store = np.vstack((flux_store, (Tij, i + 1, n_cities[j])))

    work_place = np.zeros(num_agents)
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

    for i in range(num_agents):
        home_store1[i, :] = int(all_x[i]), int(all_y[i])

    work_store = np.int64(work_store1)
    home_store = np.int64(home_store1)

    return all_x, all_y, centers, city_label, work_store, home_store


class DiseaseModel(Model):
    def __init__(self, no_people, total_area, no_agents, all_x, all_y, infection_rate, first_infected, mobility,
                 work_store, home_store):
        self.num_agents = no_agents
        grid_size = round(math.sqrt((self.num_agents / no_people) * total_area) * 100)
        self.grid = MultiGrid(grid_size, grid_size, False)
        self.schedule = RandomActivation(self)
        self.running = True

        self.stepper = 0

        for i in range(self.num_agents):
            a = Agent(i, self, infection_rate, work_store, home_store, mobility, self.stepper)
            self.schedule.add(a)
            self.grid.place_agent(a, (int(all_x[i]), int(all_y[i])))

            if i == first_infected:
                a.infected = 1

        self.datacollector = DataCollector(
            model_reporters={"Tot infections": compute_informed},
            agent_reporters={"Infected": "infected", "R-Number": "rnumber"})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        self.stepper += 1
        #print(self.stepper)


num = 2000
n = 50
Nc_N = 0.2
city_to_country = 0.14
no_people = 67000000
total_area = 240000
city_to_country_area = 13
countryside = 0.8

all_x, all_y, centers, city_label, work_store, home_store = agent_locator(city_to_country,
                                                                          no_people,
                                                                          total_area,
                                                                          city_to_country_area,
                                                                          countryside,
                                                                          num,
                                                                          n,
                                                                          Nc_N)

#############################################################################

"""model = DiseaseModel(no_people=67000000,
                     total_area=240000,
                     no_agents=num,
                     all_x=all_x,
                     all_y=all_y,
                     infection_rate=0.01,
                     first_infected=1,
                     mobility=0,
                     work_store=work_store,
                     home_store=home_store)"""

infected_total = np.zeros(0)
w = 11
w1 = 0
list2 = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1]
list1 = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1]

for i in list2:
    for j in list1:
        print(i, j, datetime.datetime.now() - begin_time)
        model = DiseaseModel(no_people=67000000,
                             total_area=240000,
                             no_agents=num,
                             all_x=all_x,
                             all_y=all_y,
                             infection_rate=i,
                             first_infected=1,
                             mobility=j,
                             work_store=work_store,
                             home_store=home_store)

        steps = len(spread_average_uk)
        for day_step in range(steps):
            model.step()

        out = model.datacollector.get_agent_vars_dataframe().groupby('Step').sum()
        new_out = out.to_numpy()[:, 0]

        infected_total = np.concatenate((infected_total, new_out), axis=None)

infected_total = np.reshape(infected_total, ((w-w1)*(w-w1), len(spread_average_uk)))
overall_count = np.zeros((w-w1)*(w-w1))
print(infected_total)

for i in range(0, (w-w1)*(w-w1)):
    countering = 0
    for j in range(len(spread_average_uk)):
        if infected_total[i, j] < 300:
            countering += 1
    overall_count[i] = countering

overall_count = np.reshape(overall_count, ((w-w1), (w-w1))).astype(int)
print(overall_count)


def plot_matrix(matrix):
    #c = mcolors.ColorConverter().to_rgb
    #rvb = make_colormap([c('black'), c('red'), 0.05, c('red'), c('yellow'), 0.5, c('yellow'), c('white')
    #                        , 0.9, c('white')])
    cax = plt.imshow(matrix, cmap=plt.get_cmap('viridis'))
    plt.xticks(np.arange(np.shape(matrix)[0]), labels=[int(i*100) for i in list2])
    plt.yticks(np.arange(np.shape(matrix)[0]), labels=list1)
    plt.xlabel('Mobility %')
    plt.ylabel('Infection Rate')
    plt.colorbar(cax)

    for i in range(np.shape(matrix)[0]):
        for j in range(np.shape(matrix)[0]):
            text = plt.text(j, i, round(matrix[i, j], 2), ha="center", va="center")

    plt.show()


plot_matrix(overall_count)

#############################################################################

"""fig, ax = plt.subplots(figsize=(10, 5))

par1 = ax.twinx()

ax.set_xlabel("Steps")
ax.set_ylabel("No. People Infected")
par1.set_ylabel("Mobility %")

p6, = par1.plot(np.arange(0, len(spread_average_uk)), spread1, color='blue', label="Mobility %")
p4, = par1.plot(np.arange(0, len(spread_average_uk1)), spread2, ':', color='dimgrey', label="Mobility %")
p5, = par1.plot(np.arange(0, len(spread_average_uk2)), spread3, ':', color='dimgrey', label="Mobility %")
p1, = ax.plot(np.arange(0, len(new_out)), new_out, color='red', label="Real Mobility")
p2, = ax.plot(np.arange(0, len(new_out1)), new_out1, color='green', label="Earlier Shift")
p3, = ax.plot(np.arange(0, len(new_out2)), new_out2, color='tab:orange', label="Later Shift")

ax.legend(handles=[p1, p2, p3], loc='lower right')

par1.yaxis.label.set_color('blue')
ax.grid(linestyle='--')

secax = ax.secondary_xaxis(-0.15, functions=(lambda x: x/8, lambda x: x/8))
secax.set_xlabel('Days')

fig.tight_layout()
plt.show()"""

#############################################################################

print(datetime.datetime.now() - begin_time)

