# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 12:51:47 2019

@author: michael.schulte
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')

LEARNING_RATE = 0.1
DISCOUNT = 0.9

EPISODES = 3000
SHOW_EVERY = 100

epsilon = 0.2
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

DISCRETE_OBS_SPACE_SIZE = [15] * len(env.observation_space.high)
discrete_obs_space_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBS_SPACE_SIZE

q_table = np.random.uniform(low = -2, high = 0,
                            size = (DISCRETE_OBS_SPACE_SIZE + [env.action_space.n]))

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_obs_space_win_size

    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
    episode_reward = 0

    if episode % SHOW_EVERY == 0:
        render = True
        #print(episode)
    else:
        render = False

    discrete_state = get_discrete_state(env.reset())

    done = False

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(low = 0, high = env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        episode_reward += reward

        new_discrete_state = get_discrete_state(new_state)

        if render == True:
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action, )] = 0
            print('we made it on episode: {}'.format(episode))

        discrete_state = new_discrete_state

        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)

    if episode % SHOW_EVERY == 0:
        average_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(np.min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(np.max(ep_rewards[-SHOW_EVERY:]))

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], color = 'r', label = 'avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], color = 'g', label = 'min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], color = 'b', label = 'max')
#plt.legend(loc = 4)
plt.show()