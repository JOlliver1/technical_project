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


class Agent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.infected = 0
        # self.mobile = mobility

    def spread_disease(self):
        if self.infected == 0:
            return

        else:
            cellmates = self.model.grid.get_cell_list_contents([self.pos])
            for a in cellmates:
                if a.infected != 1:
                    a.infected = 1

    def move(self):
        if random.uniform(0, 1) < average[day]/100:
            possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            new_position = self.random.choice(possible_steps)
            self.model.grid.move_agent(self, new_position)
        else:
            return

    def step(self):
        self.move()
        self.spread_disease()


def compute_informed(model):
    return sum([1 for a in model.schedule.agents if a.infected == 1])


class DiseaseModel(Model):
    def __init__(self, city_to_country):
        self.num_agents = 2000
        self.grid = MultiGrid(200, 200, True)
        self.schedule = RandomActivation(self)
        self.running = True

        centers = np.zeros((1, 2))
        centers[0, :] = random.randrange(10, self.grid.width - 10), random.randrange(10, self.grid.height - 10)
        x = np.zeros((1, round(int(city_to_country * self.num_agents))))
        y = np.zeros((1, round(int(city_to_country * self.num_agents))))
        x[0, :] = np.around(np.random.normal(centers[0, 0], 3, round(int(city_to_country * self.num_agents))))
        y[0, :] = np.around(np.random.normal(centers[0, 1], 3, round(int(city_to_country * self.num_agents))))

        count = 0
        while counter(x) > 2:
            runner = True
            while runner:
                new_center = (random.randrange(10, self.grid.width - 10), random.randrange(10, self.grid.height - 10))
                if dist_check(new_center, centers) & count < 20:
                    centers = np.vstack((centers, new_center))
                    runner = False

            new_x = np.around(np.random.normal(centers[count, 0], 3,
                                               round(int(city_to_country * self.num_agents) / (count + 2))))
            new_y = np.around(np.random.normal(centers[count, 1], 3,
                                               round(int(city_to_country * self.num_agents) / (count + 2))))
            while len(new_x) < round(int(city_to_country * self.num_agents)):
                new_x = np.append(new_x, -1)
                new_y = np.append(new_y, -1)

            x = np.vstack((x, new_x))
            y = np.vstack((y, new_y))
            count += 1

        new_x = np.delete(x.flatten(), np.where(x.flatten() == -1))
        new_y = np.delete(y.flatten(), np.where(y.flatten() == -1))

        x_countryside = np.around(np.random.uniform(0, self.grid.width-1, int(self.num_agents - len(new_x))))
        y_countryside = np.around(np.random.uniform(0, self.grid.height-1, int(self.num_agents - len(new_y))))

        all_x = np.concatenate((new_x, x_countryside))
        all_y = np.concatenate((new_y, y_countryside))

        for i in range(self.num_agents):
            a = Agent(i, self)
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


model = DiseaseModel(city_to_country=0.14)


def colour_plotter(model):
    c = mcolors.ColorConverter().to_rgb
    rvb = make_colormap([c('black'), c('red'), 0.05, c('red'), c('yellow'), 0.5, c('yellow'), c('white')
                        , 0.9, c('white')])

    agent_counts = np.zeros((model.grid.width, model.grid.height))

    for cell in model.grid.coord_iter():
        cell_content, x, y = cell
        agent_count = len(cell_content)
        agent_counts[x][y] = agent_count

    plt.imshow(agent_counts, interpolation='nearest', cmap=rvb)
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
            index_y[count1] = y
            count1 += 1
            if a.infected == 1:
                infected_index_x[count] = x
                infected_index_y[count] = y
                count += 1

    plt.figure(figsize=(6, 6))
    plt.scatter(index_x, index_y, marker='s', c='blue', s=2)
    plt.scatter(infected_index_x, infected_index_y, marker='s', c='red', s=2)
    plt.title('Day = ' + str(day) + '     No. Infected = ' + str(no_infected))
    plt.show()


colour_plotter(model)

#recovery_count = np.zeros(1000)

steps = 289
for day in range(steps):
    if day % 10 == 0:
        infected_plotter(model, day)
    elif day < 10:
        infected_plotter(model, day)
    model.step()

colour_plotter(model)

out = model.datacollector.get_agent_vars_dataframe().groupby('Step').sum()
#print(model.datacollector.get_agent_vars_dataframe().groupby('Step'))
#print(out)
new_out = out.to_numpy()

plt.figure(figsize=(10, 5))
plt.plot(np.arange(0, steps), new_out, color='blue', label='Real')
plt.xlabel('Days')
plt.ylabel('No. of People Infected')
plt.legend()
plt.grid()
plt.show()  # """

print(datetime.datetime.now() - begin_time)
