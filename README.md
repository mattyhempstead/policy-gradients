# policy-gradients
A repo for experiments with policy gradients in deep reinforcement learning


# To test ideas
Write a softmax network which outputs probabilities of actions which give 5, 10, 20 rewards respectively.
Try to train this network using my current ideas of how to train with policy gradients.

Start by setting the gradients for logits to be (1−S) G_t  for selected action, and (−S) G_t  for non-selected actions.
Then try a method where I simply define the loss function and Tensorflow figures out the gradients.
