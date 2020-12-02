from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid

import numpy as np
import matplotlib.pyplot as plt
import random


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
            if i < 80:
                if random.random() < 0.03:
                    a.infected = 1
            if i > 160:
                if random.random() < 0.05:
                    a.infected = 1
            else:
                if random.random() < 0.005:
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


model = News_Model(N=750,
                   width=60,
                   height=60,
                   initial=1)

steps = 200
for i in range(steps):
    model.step()

out = model.datacollector.get_agent_vars_dataframe().groupby('Step').sum()
new_out = out.to_numpy()
print(new_out)
print(np.diff(new_out.T))

plt.subplot(211)
plt.plot(np.arange(0, steps), new_out, 'o')
plt.xlabel('Steps')
plt.ylabel('No. of People')
plt.grid()
plt.subplot(212)
plt.plot(np.arange(0, steps-1), np.diff(new_out.T).T, 'x')
plt.xlabel('Steps')
plt.ylabel('No. of New People')
plt.grid()
plt.show()

