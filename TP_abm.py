from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid

import numpy as np
import matplotlib.pyplot as plt
import random

from import_apple_data import average, new_average


class Agent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.infected = 0

    def spread_news(self):
        if self.infected == 0:
            return
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=True)
        neig_agents = [a for n in neighbors for a in self.model.grid.get_cell_list_contents(n.pos)]
        for a in neig_agents:
            if random.random() < average[day]/3000:
                a.infected = 1

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def step(self):
        self.move()
        self.spread_news()


def compute_informed(model):
    return sum([1 for a in model.schedule.agents if a.infected == 1])


class News_Model(Model):
    def __init__(self, N, width, height, initial):
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.running = True

        for i in range(self.num_agents):
            a = Agent(i, self)
            self.schedule.add(a)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))
            if i < initial:
                a.infected = 1

        self.datacollector = DataCollector(
            model_reporters={"Tot informed": compute_informed},
            agent_reporters={"Infected": "infected"})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


model = News_Model(N=1000,
                   width=70,
                   height=70,
                   initial=1)

steps = 322
for day in range(steps):
    model.step()

out = model.datacollector.get_agent_vars_dataframe().groupby('Step').sum()
new_out = out.to_numpy()

plt.figure(figsize=(10, 5))
plt.plot(np.arange(0, steps), new_out, color='blue')
plt.xlabel('Steps')
plt.ylabel('No. of People')
plt.grid()
plt.show()
