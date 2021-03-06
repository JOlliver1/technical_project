from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid

import numpy as np
import matplotlib.pyplot as plt
import random
import math
import datetime

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
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def step(self):
        self.move()
        self.spread_disease()


def compute_informed(model):
    return sum([1 for a in model.schedule.agents if a.infected == 1])


class DiseaseModel(Model):
    def __init__(self, city_to_country, no_people, total_area, city_to_country_area, countryside):
        self.num_agents = 2000
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


def colour_plotter(model):

    agent_counts = np.zeros((model.grid.width, model.grid.height))

    for cell in model.grid.coord_iter():
        cell_content, x, y = cell
        agent_count = len(cell_content)
        agent_counts[x][y] = agent_count

    plt.imshow(agent_counts, interpolation='nearest', cmap='hot')
    plt.colorbar()
    plt.show()


#colour_plotter(model)

#recovery_count = np.zeros(1000)

steps = 200
for day in range(steps):
    model.step()

#colour_plotter(model)

out = model.datacollector.get_agent_vars_dataframe().groupby('Step').sum()
new_out = out.to_numpy()

"""plt.figure(figsize=(10, 5))
plt.plot(np.arange(0, steps), new_out, color='blue', label='Real')
plt.xlabel('Days')
plt.ylabel('No. of People Infected')
plt.legend()
plt.grid()
plt.show()"""

print(datetime.datetime.now() - begin_time)

