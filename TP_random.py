from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
# to collect features during the #simulation
from mesa.space import MultiGrid
# to generate the environment


# for computation and visualization purpose
import numpy as np
import sys
import matplotlib.pyplot as plt
import random


class Agent(Agent):
    """ An agent with fixed initial wealth."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.knowledge = 0

    def spread_news(self):
        if self.knowledge == 0:
            return
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=True)
        neig_agents = [a for n in neighbors for a in self.model.grid.get_cell_list_contents(n.pos)]
        for a in neig_agents:
            if random.random() < 0.3:
                a.knowledge = 1

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


# let's define a function which is able to count, at each step, how many agents
# are aware of the product.

def compute_informed(model):
    return sum([1 for a in model.schedule.agents if a.knowledge == 1])

# now let's define the model


class News_Model(Model):
    def __init__(self, N, width, height):
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            a = Agent(i, self)
            self.schedule.add(a)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))
            l = [1, 2, 3, 4, 5]
            # 5 agents are aware of the product
            if i in l:  # only agents with id in the list are aware of the product
                a.knowledge = 1

        self.datacollector = DataCollector(
            model_reporters={"Tot informed": compute_informed},
            agent_reporters={"Knowledge": "knowledge"})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


'''Run the model '''
model = News_Model(100, 50, 50)
for i in range(100):
    model.step()

# let's inspect the results:
out = model.datacollector.get_agent_vars_dataframe().groupby('Step').sum()
new_out = out.to_numpy()
#print(type(new_out[:]))

plt.plot(np.arange(0, 100), new_out[:])
plt.show()

