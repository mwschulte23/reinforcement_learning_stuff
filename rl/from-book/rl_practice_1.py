# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 19:22:32 2019

@author: michael.schulte
"""

import random

class Environment:
    def __init__(self):
        self.steps_left = 10


    def get_observations(self):
        return [0.0, 0.0, 0.0]


    def get_actions(self):
        return [0, 1]


    def is_done(self):
        return self.steps_left == 0


    def action(self, action):
        if self.is_done():
            raise Exception('game is over')

        self.steps_left -= 1

        return random.random()


class Agent:
    def __init__(self):
        self.total_reward = 0


    def step(self, env):
        current_obs = env.get_observations()
        actions = env.get_actions()
        reward = env.action(random.choice(actions))
        self.total_reward += reward


if __name__ == '__main__':
    env = Environment()
    agent = Agent()

    while not env.is_done():
        agent.step(env)

    print('Total reward = {}'.format(agent.total_reward))






