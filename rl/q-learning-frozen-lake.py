# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 13:36:33 2019

@author: michael.schulte
"""

import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

alpha = 0.2
alpha_decay = .99999

GAMMA = .9

epsilon = 0.5
eps_decay = .999

EPISODES = 1000
SHOW_EVERY = 10000

reward_tracking = []

q_table = np.random.uniform(low = 0, high = 1,
                            size = ([env.observation_space.n] + [env.action_space.n]))

alpha_trend = []
eps_trend = []

for ep in range(EPISODES):
    episode_reward = 0
    state = env.reset()

    if ep % SHOW_EVERY == 0:
        render = False
    else:
        render = False

    done = False

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[state])
        else:
            action = np.random.randint(low = 0, high = env.action_space.n)

        new_state, reward, done, _ = env.step(action)

        episode_reward += reward

        if render:
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_state])
            current_q = q_table[state, action]
            new_q = (1 - alpha) * current_q + alpha * (reward + GAMMA * max_future_q)
            q_table[state, action] = new_q
        elif new_state == 15:
            q_table[state, action] = 1
#            print('we won on episode: {}'.format(ep))

        state = new_state

    reward_tracking.append(episode_reward)

    avg_n_reward = np.mean(reward_tracking[-100:])

    if avg_n_reward < .1:
        epsilon = 0.5

    epsilon *= eps_decay
    eps_trend.append(epsilon)

    if avg_n_reward > .7:
        alpha = .05

    alpha *= alpha_decay
    alpha_trend.append(alpha)

    if avg_n_reward > np.mean(reward_tracking[-200:-100]):
        print('Last 20 Episodes Improved to {}'.format(avg_n_reward))

env.close()


fig, ax = plt.subplots(3, 1, figsize = (6, 6))

ax[0].plot(pd.Series(reward_tracking).rolling(100).mean())
ax[1].plot(pd.Series(eps_trend).rolling(1).mean())
ax[2].plot(pd.Series(alpha_trend).rolling(1).mean())
plt.show()
