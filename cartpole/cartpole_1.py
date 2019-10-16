"""
This script trains an agent for cartpole using MC policy gradients
The agent is able to solve after ~200 generations however it will often forget its knowledge
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


TRAINING_EPOCHS = 1000  
DISCOUNT_FACTOR = 0.9


# Plot total reward over time
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
line, = ax.plot(total_rewards)


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
            ax.set_xlim(0, len(total_rewards))
            ax.set_ylim(0, max(total_rewards))
            line.set_ydata(total_rewards)
            line.set_xdata(list(range(len(total_rewards))))
            fig.canvas.draw()
            fig.canvas.flush_events()

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
plt.plot(total_rewards)
plt.show()
