from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid

import numpy as np
import matplotlib.pyplot as plt
import random
import math

from import_apple_data import average


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

    for i in range(np.shape(array)[0]):
        for j in range(len(array[i, :])):
            if array[i, j] != -1:
                count += 1

    return count


class Agent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.infected = 0
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
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def step(self):
        self.move()
        self.spread_disease()


def compute_informed(model):
    return sum([1 for a in model.schedule.agents if a.infected == 1])


class DiseaseModel(Model):
    def __init__(self, city_ratio, city_to_country):
        self.num_agents = 2000
        self.grid = MultiGrid(200, 200, True)
        self.schedule = RandomActivation(self)
        self.running = True

        centers = np.zeros((1, 2))
        centers[0, :] = random.randrange(20, self.grid.width - 20), random.randrange(20, self.grid.height - 20)
        x = np.zeros((1, round(int(city_to_country * self.num_agents))))
        y = np.zeros((1, round(int(city_to_country * self.num_agents))))
        x[0, :] = np.around(np.random.normal(centers[0, 0], 5, round(int(city_to_country * self.num_agents))))
        y[0, :] = np.around(np.random.normal(centers[0, 1], 5, round(int(city_to_country * self.num_agents))))

        count = 0
        while counter(x) < self.num_agents - ((1-city_ratio)*self.num_agents):
            runner = True
            while runner:
                new_center = (random.randrange(20, 180), random.randrange(20, 180))
                if dist_check(new_center, centers):
                    centers = np.vstack((centers, new_center))
                    runner = False

            new_x = np.around(np.random.normal(centers[count, 0], 5,
                                               round(int(city_to_country * self.num_agents) / ((count + 2) ** 1.07))))
            new_y = np.around(np.random.normal(centers[count, 1], 5,
                                               round(int(city_to_country * self.num_agents) / ((count + 2) ** 1.07))))
            while len(new_x) < round(int(city_to_country * self.num_agents)):
                new_x = np.append(new_x, -1)
                new_y = np.append(new_y, -1)

            x = np.vstack((x, new_x))
            y = np.vstack((y, new_y))
            count += 1

        x_countryside = np.around(np.random.uniform(0, self.grid.width-1, int((1 - city_ratio) * self.num_agents)))
        y_countryside = np.around(np.random.uniform(0, self.grid.height-1, int((1 - city_ratio) * self.num_agents)))

        all_x = np.concatenate((x.flatten(), x_countryside))
        all_y = np.concatenate((y.flatten(), y_countryside))
        new_all_x = np.delete(all_x, np.where(all_x == -1))
        new_all_y = np.delete(all_y, np.where(all_y == -1))

        for i in range(self.num_agents):
            a = Agent(i, self)
            self.schedule.add(a)
            self.grid.place_agent(a, (int(new_all_x[i]), int(new_all_y[i])))

            if i < 1:
                a.infected = 1

        self.datacollector = DataCollector(
            model_reporters={"Tot informed": compute_informed},
            agent_reporters={"Infected": "infected"})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


model = DiseaseModel(city_ratio=0.5, city_to_country=0.14)


def colour_plotter(model):

    agent_counts = np.zeros((model.grid.width, model.grid.height))

    for cell in model.grid.coord_iter():
        cell_content, x, y = cell
        agent_count = len(cell_content)
        agent_counts[x][y] = agent_count

    plt.imshow(agent_counts, interpolation='nearest', cmap='hot')
    plt.colorbar()
    plt.show()


colour_plotter(model)

#recovery_count = np.zeros(1000)

steps = 289
for day in range(steps):
    model.step()

#colour_plotter(model)

out = model.datacollector.get_agent_vars_dataframe().groupby('Step').sum()
new_out = out.to_numpy()

plt.figure(figsize=(10, 5))
plt.plot(np.arange(0, steps), new_out, color='blue', label='Real')
plt.xlabel('Days')
plt.ylabel('No. of People Infected')
plt.legend()
plt.grid()
plt.show()
