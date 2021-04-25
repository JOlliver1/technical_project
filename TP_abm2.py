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
    #plt.title('Distribution of ' + str(model.num_agents) + ' Agents for the UK')
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


def infected_plotter1(model, day, centers1):

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

    for i in centers1:
        i[1] = model.grid.height - i[1]

    city_1 = centers1[0, :]
    city_2 = centers1[1, :]
    radius = math.sqrt((city_1[0] - city_2[0]) ** 2 + (city_1[1] - city_2[1]) ** 2)
    circle = plt.Circle((city_1[0], city_1[1]), radius, color='black', fill=False)
    circle1 = plt.Circle((city_1[0], city_1[1]), 10, color='black', fill=False)
    circle2 = plt.Circle((city_2[0], city_2[1]), 6, color='black', fill=False)

    #print(city_1[0], city_1[1])
    plt.rcParams['axes.facecolor'] = 'white'
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), squeeze=False)
    ax[0, 0].scatter(index_x, index_y, marker='s', c='blue', s=0.5)
    ax[0, 0].scatter(infected_index_x, infected_index_y, marker='s', c='red', s=0.5)
    ax[0, 0].arrow(city_1[0], city_1[1], city_2[0]-city_1[0], city_2[1]-city_1[1], fc="k", ec="k", head_width=5,
              head_length=10, width=0.7, length_includes_head=True)
    ax[0, 0].add_patch(circle)
    ax[0, 0].add_patch(circle1)
    ax[0, 0].add_patch(circle2)
    #plt.title('Step = ' + str(day) + '     No. Infected = ' + str(no_infected) + '/' + str(model.num_agents))
    ax[0, 0].axes.get_xaxis().set_visible(False)
    ax[0, 0].axes.get_yaxis().set_visible(False)
    plt.show()


def infected_calc(model):

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

    return index_x, index_y, infected_index_x, infected_index_y, no_infected


class Agent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.infected = 0

    def spread_disease(self):
        if self.infected == 0:
            return

        else:
            cellmates = self.model.grid.get_cell_list_contents([self.pos])
            for a in cellmates:
                if a.infected != 1:
                    a.infected = 1

    def move(self):
        if 10 < 20:  # random.uniform(0, 1) < average_uk[day]/100:
            possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            new_position = self.random.choice(possible_steps)
            self.model.grid.move_agent(self, new_position)
            #print(type(new_position), new_position)
        else:
            return

    def step(self):
        self.move()
        self.spread_disease()


def compute_informed(model):
    return sum([1 for a in model.schedule.agents if a.infected == 1])


class DiseaseModel(Model):
    def __init__(self, city_to_country, no_people, total_area, city_to_country_area, countryside):
        self.num_agents = 2000
        grid_size = round(math.sqrt((self.num_agents/no_people)*total_area)*100)
        self.grid = MultiGrid(grid_size, grid_size, False)
        self.schedule = RandomActivation(self)
        self.running = True

        global centers
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

            new_x = np.around(np.random.normal(centers[count, 0], (1/(6*city_to_country_area*(math.sqrt(count+1))))
                                               * self.grid.width, round(int(city_to_country * self.num_agents)
                                                                        / (count + 2))))
            new_y = np.around(np.random.normal(centers[count, 1], (1/(6*city_to_country_area*(math.sqrt(count+1))))
                                               * self.grid.height, round(int(city_to_country * self.num_agents)
                                                                         / (count + 2))))
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


model = DiseaseModel(city_to_country=0.14,
                     no_people=67000000,
                     total_area=240000,
                     city_to_country_area=13,
                     countryside=0.8)

infected_plotter1(model, 0, centers)

#print(centers)

#colour_plotter(model)

#recovery_count = np.zeros(1000)

"""steps = 289
for day in range(steps):
    if day % 10 == 0:
        infected_plotter(model, day)
    elif day < 10:
        infected_plotter(model, day)
    model.step()"""
"""
steps = 289
for day in range(steps):
    if day == 0:
        index_x, index_y, infected_index_x, infected_index_y, no_infected = infected_calc(model)
    if day == 5:
        index_x1, index_y1, infected_index_x1, infected_index_y1, no_infected1 = infected_calc(model)
    if day == 10:
        index_x5, index_y5, infected_index_x5, infected_index_y5, no_infected5 = infected_calc(model)
    if day == 50:
        index_x10, index_y10, infected_index_x10, infected_index_y10, no_infected10 = infected_calc(model)
    if day == 100:
        index_x50, index_y50, infected_index_x50, infected_index_y50, no_infected50 = infected_calc(model)
    if day == 200:
        index_x100, index_y100, infected_index_x100, infected_index_y100, no_infected100 = infected_calc(model)
    model.step()

plt.rcParams['axes.facecolor'] = 'black'
#plt.figure(figsize=(12, 6))
fig, ax = plt.subplots(2, 3, figsize=(10, 7))
#fig.figure(figsize=(10, 10))

#plt.subplot(3, 3, 1)
ax[0, 0].scatter(index_x, index_y, marker='s', c='blue', s=0.5)
ax[0, 0].scatter(infected_index_x, infected_index_y, marker='s', c='red', s=0.5)
ax[0, 0].title.set_text('Step=' + str(0) + '     No. Infected=' + str(no_infected) + '/' + str(2000))
ax[0, 0].axes.get_xaxis().set_visible(False)
ax[0, 0].axes.get_yaxis().set_visible(False)

#plt.subplot(3, 3, 2)
ax[0, 1].scatter(index_x1, index_y1, marker='s', c='blue', s=0.5)
ax[0, 1].scatter(infected_index_x1, infected_index_y1, marker='s', c='red', s=0.5)
ax[0, 1].title.set_text('Step=' + str(5) + '     No. Infected=' + str(no_infected1) + '/' + str(2000))
ax[0, 1].axes.get_xaxis().set_visible(False)
ax[0, 1].axes.get_yaxis().set_visible(False)

#plt.subplot(3, 3, 3)
ax[0, 2].scatter(index_x5, index_y5, marker='s', c='blue', s=0.5)
ax[0, 2].scatter(infected_index_x5, infected_index_y5, marker='s', c='red', s=0.5)
ax[0, 2].title.set_text('Step=' + str(10) + '     No. Infected=' + str(no_infected5) + '/' + str(2000))
ax[0, 2].axes.get_xaxis().set_visible(False)
ax[0, 2].axes.get_yaxis().set_visible(False)

#plt.subplot(3, 3, 4)
ax[1, 0].scatter(index_x10, index_y10, marker='s', c='blue', s=0.5)
ax[1, 0].scatter(infected_index_x10, infected_index_y10, marker='s', c='red', s=0.5)
ax[1, 0].title.set_text('Step=' + str(50) + '     No. Infected=' + str(no_infected10) + '/' + str(2000))
ax[1, 0].axes.get_xaxis().set_visible(False)
ax[1, 0].axes.get_yaxis().set_visible(False)

#plt.subplot(3, 3, 5)
ax[1, 1].scatter(index_x50, index_y50, marker='s', c='blue', s=0.5)
ax[1, 1].scatter(infected_index_x50, infected_index_y50, marker='s', c='red', s=0.5)
ax[1, 1].title.set_text('Step=' + str(100) + '     No. Infected=' + str(no_infected50) + '/' + str(2000))
ax[1, 1].axes.get_xaxis().set_visible(False)
ax[1, 1].axes.get_yaxis().set_visible(False)

#plt.subplot(3, 3, 6)
ax[1, 2].scatter(index_x100, index_y100, marker='s', c='blue', s=0.5)
ax[1, 2].scatter(infected_index_x100, infected_index_y100, marker='s', c='red', s=0.5)
ax[1, 2].title.set_text('Step=' + str(200) + '     No. Infected=' + str(no_infected100) + '/' + str(2000))
ax[1, 2].axes.get_xaxis().set_visible(False)
ax[1, 2].axes.get_yaxis().set_visible(False)

fig.tight_layout()
plt.show()"""

"""colour_plotter(model)

out = model.datacollector.get_agent_vars_dataframe().groupby('Step').sum()
#print(model.datacollector.get_agent_vars_dataframe().groupby('Step'))
#print(out)
new_out = out.to_numpy()

plt.rcParams['axes.facecolor'] = 'white'
plt.figure(figsize=(12, 6))
plt.plot(np.arange(0, steps), new_out, color='blue', label='Real')
plt.xlabel('Steps')
plt.ylabel('No. of People Infected')
plt.legend()
plt.grid()
plt.show()  # """

print(datetime.datetime.now() - begin_time)
