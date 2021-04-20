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

from import_apple_data import average

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

    work_store2 = work_store
    print(work_store2)

    plt.rcParams['axes.facecolor'] = 'black'
    plt.figure(figsize=(6, 6))
    plt.scatter(index_x, index_y, marker='s', c='blue', s=0.5)
    plt.scatter(infected_index_x, infected_index_y, marker='s', c='red', s=0.5)
    plt.title('Step = ' + str(day) + '     No. Infected = ' + str(no_infected) + '/' + str(model.num_agents))
    plt.show()


class Agent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.infected = 0
        self.working = 0
        # self.mobile = mobility

    def spread_disease(self):
        if self.infected == 0:
            return

        #print(self.mobile[day]/3000)
        else:
            cellmates = self.model.grid.get_cell_list_contents([self.pos])
            for a in cellmates:
                if a.infected != 1:
                    a.infected = 1

    def move(self):
        if (day % 4) - 2 == 0:
            if work_store[self.unique_id, 0] != 0:
                new_position = (tuple(work_store[self.unique_id, :]))
                self.model.grid.move_agent(self, new_position)
                self.working = 1

        if (day % 4) == 0:
            if work_store[self.unique_id, 0] != 0:
                new_position = (tuple(home_store[self.unique_id, :]))
                self.model.grid.move_agent(self, new_position)
                self.working = 0
        else:
            """possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            new_position = self.random.choice(possible_steps)
            self.model.grid.move_agent(self, new_position)
        #else:"""
            possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            while True:
                if self.working == 0:
                    new_position = self.random.choice(possible_steps)
                    if find_dist(new_position, home_store[self.unique_id, :]) <= 5:
                        self.model.grid.move_agent(self, new_position)
                        break
                elif self.working == 1:
                    new_position = self.random.choice(possible_steps)
                    if find_dist(new_position, work_store[self.unique_id, :]) <= 5:
                        self.model.grid.move_agent(self, new_position)
                        break

    def step(self):
        self.move()
        self.spread_disease()


def compute_informed(model):
    return sum([1 for a in model.schedule.agents if a.infected == 1])


class DiseaseModel(Model):
    def __init__(self, city_to_country, no_people, total_area, city_to_country_area, countryside, no_agents, Nc_N):
        self.num_agents = no_agents
        grid_size = round(math.sqrt((self.num_agents / no_people) * total_area) * 100)
        self.grid = MultiGrid(grid_size, grid_size, False)
        self.schedule = RandomActivation(self)
        self.running = True

        centers = np.zeros((1, 2))
        centers[0, :] = random.randrange(10, self.grid.width - 10), random.randrange(10, self.grid.height - 10)
        x = np.zeros((1, round(int(city_to_country * self.num_agents))))
        y = np.zeros((1, round(int(city_to_country * self.num_agents))))
        x[0, :] = np.around(np.random.normal(centers[0, 0], 3, round(int(city_to_country * self.num_agents))))
        y[0, :] = np.around(np.random.normal(centers[0, 1], 3, round(int(city_to_country * self.num_agents))))

        count = 0
        countryside_count = 0
        while countryside_count < (countryside * self.num_agents):
            countryside_count += counter(x)
            runner = True
            while runner:
                new_center = (random.randrange(10, self.grid.width - 10), random.randrange(10, self.grid.height - 10))
                if dist_check(new_center, centers):
                    centers = np.vstack((centers, new_center))
                    runner = False

            new_x = np.around(
                np.random.normal(centers[count, 0], (1 / (6 * city_to_country_area * (math.sqrt(count + 1))))
                                 * self.grid.width, round(int(city_to_country * self.num_agents)
                                                          / (count + 2))))
            new_y = np.around(
                np.random.normal(centers[count, 1], (1 / (6 * city_to_country_area * (math.sqrt(count + 1))))
                                 * self.grid.height, round(int(city_to_country * self.num_agents)
                                                           / (count + 2))))
            while len(new_x) < round(int(city_to_country * self.num_agents)):
                new_x = np.append(new_x, -1)
                new_y = np.append(new_y, -1)

            x = np.vstack((x, new_x))
            y = np.vstack((y, new_y))
            count += 1

        label = city_labeler(x)
        for i in range(len(label)):
            city_label[i] = label[i]

        new_x = np.delete(x.flatten(), np.where(x.flatten() == -1))
        new_y = np.delete(y.flatten(), np.where(y.flatten() == -1))

        x_countryside = np.around(np.random.uniform(0, self.grid.width - 1, int(self.num_agents - len(new_x))))
        y_countryside = np.around(np.random.uniform(0, self.grid.height - 1, int(self.num_agents - len(new_y))))

        all_x = np.concatenate((new_x, x_countryside))
        all_y = np.concatenate((new_y, y_countryside))

        for i in range(self.num_agents):
            a = Agent(i, self)
            self.schedule.add(a)
            self.grid.place_agent(a, (int(all_x[i]), int(all_y[i])))
            home_store1[i, :] = int(all_x[i]), int(all_y[i])

            if i == 1:
                a.infected = 1
                #a.working = 1

        n = 10
        flux_store = np.zeros((1, 3))

        for i in range(round(len(centers) / 2)):
            n_cities = random.sample(range(1, round(len(centers) / 3)), n)

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

                if Tij > 1 and (i != n_cities[j]):  # increase this and plot arrows
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

        global work_store, home_store
        work_store = np.int64(work_store1)
        home_store = np.int64(home_store1)

        self.datacollector = DataCollector(
            model_reporters={"Tot informed": compute_informed},
            agent_reporters={"Infected": "infected"})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


num = 2000
home_store1 = np.zeros((num, 2))
city_label = np.zeros(num)
#work_store = np.zeros((num, 2))
model = DiseaseModel(city_to_country=0.14,
                     no_people=67000000,
                     total_area=240000,
                     city_to_country_area=13,
                     countryside=0.8,
                     no_agents=num,
                     Nc_N=0.2)

#colour_plotter(model)

#recovery_count = np.zeros(1000)
infected_plotter(model, 0)

"""steps = 500
for day in range(steps):
    if day % 10 == 0:
        infected_plotter(model, day)
    elif day < 20:
        infected_plotter(model, day)
    model.step()"""

#colour_plotter(model)

"""out = model.datacollector.get_agent_vars_dataframe().groupby('Step').sum()
new_out = out.to_numpy()

plt.rcParams['axes.facecolor'] = 'white'
plt.figure(figsize=(12, 6))
plt.plot(np.arange(0, steps), new_out, color='blue', label='Real')
plt.xlabel('Days')
plt.ylabel('No. of People Infected')
plt.legend()
plt.grid()
plt.show()  # """

print(datetime.datetime.now() - begin_time)

