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
        while True:
            new_position = self.random.choice(possible_steps)
            if find_dist(new_position, home_store[self.unique_id, :]) <= 5:
                self.model.grid.move_agent(self, new_position)
                break

    def step(self):
        self.move()
        self.spread_disease()


def compute_informed(model):
    return sum([1 for a in model.schedule.agents if a.infected == 1])


class DiseaseModel(Model):
    def __init__(self, home_store):
        self.num_agents = 1000
        self.grid = MultiGrid(200, 200, True)
        self.schedule = RandomActivation(self)
        self.running = True

        for i in range(self.num_agents):
            a = Agent(i, self)
            self.schedule.add(a)
            while True:
                #x = round(int(np.random.normal(self.grid.width/2, 10, 1)))
                #y = round(int(np.random.normal(self.grid.height/2, 10, 1)))
                x = self.random.randrange(self.grid.width)
                y = self.random.randrange(self.grid.height)
                if len(self.grid.get_neighbors((x, y), moore=True, include_center=True, radius=10)) <= 7:
                    self.grid.place_agent(a, (x, y))
                    home_store[i, :] = x, y
                    break

            if i < 1:
                a.infected = 1

        self.datacollector = DataCollector(
            model_reporters={"Tot informed": compute_informed},
            agent_reporters={"Infected": "infected"})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


home_store = np.zeros((1000, 2))
model = DiseaseModel(home_store=home_store)


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
