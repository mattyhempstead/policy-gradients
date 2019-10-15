import tensorflow as tf
from tensorflow import keras

def loss(actions_and_rewards, logits):
    # Below is an explanation of the loss function

    # First split the actions/rewards into two 1D arrays
    # This assumes they are passed in as action-reward pairs (e.g. [[action_1, reward_1], [action_2, reward_2]])
    # As rewards must allow for floats, actions must arrive as floats and then be casted to ints
    actions, rewards = tf.split(actions_and_rewards, 2, axis=-1)
    actions = tf.squeeze(actions, 1)
    actions = tf.cast(actions, tf.int32)
    rewards = tf.squeeze(rewards, 1)
    # print('actions and rewards', actions, rewards)

    # We begin by taking `tf.nn.sparse_softmax_cross_entropy_with_logits` which requires logits and a label
    # The label is simply the index of the correct action
    # The logits are those used to calculate the softmax probabilites for each action
    # Cross entropy gets the sum of -q_i * ln(p_i) for each variable i, where q is the true distribution and p is the predicted distribution
    # Sparse cross entropy means the true distribution consists of all zeros, with a single correct label as one (e.g. [0,0,0,1,0,0])
    # Thus, instead of passing in a whole true distribution, only the index of the correct action is needed
    # Have a sparse true distribution also means the actual value of the function is reduced to -log(p_i) where i is the correct label
    neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = actions, logits = logits)
    # print(neg_log_prob)

    # The -log(p_i) is then scaled to equal `-reward * log(p_i)`
    # This is now equivalent to maximising the policy gradient formula `reward * log(p_i)` 
    scaled_neg_log_prob = neg_log_prob * rewards

    # Lastly the loss is returned as the mean of this scaled_neg_log_prob for all actions
    return tf.reduce_mean(scaled_neg_log_prob)
