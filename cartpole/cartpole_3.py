"""
This script trains an agent for cartpole
- Uses MC policy gradients
- Normalises rewards
- Plays a batch of games before training to ensure more stable learning
  This will reduce the likelihood of the agent reinforcing bad behaviours due to a unlucky run as
  multiple runs will be averaged.
- Also limit the number of samples used for any single train, instead of just using all of those
  generated in the most recent batch of games.
  When this is not used, the agent gets far more training samples when it performs well
  This seems to cause the agent to become worse directly after it performs well, probably since
  the policy is changing faster as it approaches a maximum, when really it should slow down.

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
DISCOUNT_FACTOR = 0.95
GAME_BATCH_SIZE = 8

# Maximum number of (S_t, A_t, G_t) samples used during a single train
TRAINING_SUBSET_SIZE = 128 

# The batch size and number of epochs used in model.fit()
TRAINING_BATCH_SIZE = 8
TRAINING_EPOCH_SIZE = 1


# Plot total reward over time
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
line, = ax.plot([], "c")
line_ma, = ax.plot([], "b")

def update_plot(y_values):
    # Set new axis limits
    ax.set_xlim(0, len(y_values)-1)
    ax.set_ylim(0, max(y_values))

    # Plot y values directly
    line.set_ydata(y_values)
    line.set_xdata(list(range(len(y_values))))

    # Plot a moving average of size 10
    y_values_ma = [np.array(y_values[-10-i:][:10]).mean() for i in range(len(y_values))][::-1]
    line_ma.set_ydata(y_values_ma)
    line_ma.set_xdata(list(range(len(y_values_ma))))

    fig.canvas.draw()
    fig.canvas.flush_events()


for episode in range(TRAINING_EPOCHS):
    # print("Train No.{}".format(episode))
    states = []
    actions = []
    rewards = []

    for game_num in range(GAME_BATCH_SIZE):
        # print("Batch Game No.{}".format(game_num))

        state = env.reset()
        single_game_rewards = []

        for t in range(200):
            # env.render()

            states.append(state)

            # Get action from model policy
            state = np.array([state])
            output = model.predict(state)
            action = model.select_action(output).numpy()
            actions.append(action)

            state, reward, done, info = env.step(action)

            single_game_rewards.append(reward)

            if done:
                print("Episode finished after {} timesteps".format(t+1))

                # Update plot live
                total_rewards.append(t+1)
                update_plot(total_rewards)

                break


        # After a full game of observations, calculate G_t for each (S_t, A_t) pair
        # In this case, G_t is the total future discounted reward
        for i in range(len(single_game_rewards)-2, -1, -1):
            single_game_rewards[i] += single_game_rewards[i+1] * DISCOUNT_FACTOR

        rewards += single_game_rewards

    # print(rewards)

    # Normalise rewards
    rewards = np.array(rewards)
    rewards = (rewards - rewards.mean()) / rewards.std()

    # Generate array of inputs (S_t) and outputs (A_t, G_t) 
    inputs = np.array(states, dtype='float32')
    outputs = np.array([[actions[i], rewards[i]] for i in range(len(states))], dtype='float32')

    # Select a subset of the training data from these games
    training_subset = np.random.choice(list(range(len(inputs))), min(len(inputs), 128))
    inputs = inputs[training_subset]
    outputs = outputs[training_subset]

    # print(inputs, outputs)

    train_results = model.model.fit(
        x = inputs, 
        y = outputs,
        batch_size = TRAINING_BATCH_SIZE,
        epochs = TRAINING_EPOCH_SIZE,
    )


env.close()

plt.ioff()
update_plot(total_rewards)
plt.show()
