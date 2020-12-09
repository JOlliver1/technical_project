from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid

import numpy as np
import matplotlib.pyplot as plt
import random

from import_apple_data import average, new_average, newer_average


class Agent(Agent):
    def __init__(self, unique_id, mobility, model):
        super().__init__(unique_id, model)
        self.infected = 0
        self.mobile = mobility

    def spread_disease(self):
        if self.infected == 0:
            return
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=True)
        neig_agents = [a for n in neighbors for a in self.model.grid.get_cell_list_contents(n.pos)]
        for a in neig_agents:
            if random.random() < self.mobile[day]/3000:
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
        self.spread_disease()


def compute_informed(model):
    return sum([1 for a in model.schedule.agents if a.infected == 1])


class Disease_Model(Model):
    def __init__(self, mobility):
        self.num_agents = 1000
        self.grid = MultiGrid(100, 100, True)
        self.schedule = RandomActivation(self)
        self.running = True

        for i in range(self.num_agents):
            a = Agent(i, mobility, self)
            self.schedule.add(a)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))
            if i < 1:
                a.infected = 1

        self.datacollector = DataCollector(
            model_reporters={"Tot informed": compute_informed},
            agent_reporters={"Infected": "infected"})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


model = Disease_Model(mobility=average)

model2 = Disease_Model(mobility=new_average)

model3 = Disease_Model(mobility=newer_average)

steps = 289
for day in range(steps):
    model.step()
    model2.step()
    model3.step()

out = model.datacollector.get_agent_vars_dataframe().groupby('Step').sum()
out2 = model2.datacollector.get_agent_vars_dataframe().groupby('Step').sum()
out3 = model3.datacollector.get_agent_vars_dataframe().groupby('Step').sum()
new_out = out.to_numpy()
new_out2 = out2.to_numpy()
new_out3 = out3.to_numpy()

plt.figure(figsize=(10, 5))
plt.plot(np.arange(0, steps), new_out2, color='red', label='Behind')
plt.plot(np.arange(0, steps), new_out3, color='green', label='Ahead')
plt.plot(np.arange(0, steps), new_out, color='blue', label='Real')
plt.xlabel('Days')
plt.ylabel('No. of People')
plt.legend()
plt.grid()
plt.show()
