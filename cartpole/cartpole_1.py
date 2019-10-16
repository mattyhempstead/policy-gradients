"""
This script trains an agent for cartpole using MC policy gradients
The agent is able to solve after a few hundred generations however it will often forget its knowledge
and oscilate between non-solved and solved states.
"""

import math, os, sys, random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import gym
import matplotlib.pyplot as plt

from model import Model

# Stop some of the random logging
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


model = Model()

env = gym.make('CartPole-v0')

total_rewards = []


TRAINING_EPOCHS = 3000
DISCOUNT_FACTOR = 0.9


# Plot total reward over time
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
line, = ax.plot([], "c")
line_ma, = ax.plot([], "b")

def update_plot(y_values):
    ax.set_xlim(0, len(y_values)-1)
    ax.set_ylim(0, max(y_values))

    line.set_ydata(y_values)
    line.set_xdata(list(range(len(y_values))))

    # Plot a moving average of size 10
    y_values_ma = [np.array(y_values[-10-i:][:10]).mean() for i in range(len(y_values))][::-1]
    line_ma.set_ydata(y_values_ma)
    line_ma.set_xdata(list(range(len(y_values_ma))))

    fig.canvas.draw()
    fig.canvas.flush_events()

for episode in range(TRAINING_EPOCHS):

    states = []
    actions = []
    rewards = []

    state = env.reset()

    for t in range(200):
        # env.render()
        # print(observation)

        states.append(state)

        state = np.array([state])
        output = model.predict(state)
        action = model.select_action(output).numpy()
        actions.append(action)

        state, reward, done, info = env.step(action)

        rewards.append(reward)

        # print(reward)

        if done:
            print("Episode finished after {} timesteps".format(t+1))

            # Update plot live
            total_rewards.append(t+1)
            update_plot(total_rewards)

            break


    # After a full game of observations, calculate G_t for each (S_t, A_t) pair
    # In this case, G_t is the total future discounted reward
    for i in range(len(rewards)-2, -1, -1):
        rewards[i] += rewards[i+1] * DISCOUNT_FACTOR

    # print(actions, rewards)

    # Generate array of inputs (S_t) and outputs (A_t, G_t) 
    inputs = np.array(states, dtype='float32')
    outputs = np.array([[actions[i], rewards[i]] for i in range(len(states))], dtype='float32')

    # print(inputs, outputs)


    train_results = model.model.fit(
        x = inputs, 
        y = outputs,
        batch_size = 2,
        epochs = 1,
    )


env.close()

plt.ioff()
update_plot(total_rewards)
plt.show()
