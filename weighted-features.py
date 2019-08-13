"""
This program successfully demonstrates the converging properties of the policy gradient algorithm
Given N actions, each with different reward values, and algorithm is able to correctly determine which 
is worth the most reward and converge to this action.

This true for all cases, even if the algorithm begins highly biased towards a lower reward rating,
it will still converge to the correct action with a small enough learning rate.
In practice, however, the algorithm often has trouble converging to higher rewards when an action with 
similar reward already has a high probability and the best action has a low probability to being with.
"""

import numpy as np, random, math

print('Testing weighted features')

LEARNING_RATE = 0.001
FEATURES = 2
REWARDS = np.array([10, 11])
OUTPUTS = len(REWARDS)

weights = np.array([[random.random() for i in range(FEATURES)] for j in range(OUTPUTS)])
weights = np.array([
    np.array([0.0,5.0]),
    np.array([0.0,5.0])
])

print('Rewards', REWARDS)
print(weights)

def getOutput(weights, features):
    output = weights.dot(features)
    output = np.exp(output)
    output = output / sum(output)
    return {
        "output": output,
        "choice": np.random.choice(list(range(len(output))), p=output)
    }

def train(weights, output):
    choice = output["choice"]

    # Calculate policy gradients
    grad = np.empty(len(REWARDS))
    for i in range(len(REWARDS)):
        if i == choice:
            grad[i] = 1 - output["output"][i]
        else:
            grad[i] = -output["output"][i]

    grad *= REWARDS[choice]
    # print(choice, grad)

    # Now actually updated weights based on policy gradients
    # dL/dW = dL/dl * dl/dw = dL/dl = grad
    for i in range(OUTPUTS):
        for w in range(FEATURES):
            weights[i][w] += LEARNING_RATE * grad[i]

    return weights


print("\nTraining")

while True:
    features = np.array([1, 1])
    output = getOutput(weights, features)
    # print(output["choice"], end=" ")
    # print(output["output"])
    for i in output["output"]:
        # length = math.floor(i * 10)
        # print("[{}] ".format("#"*length + " "*(10-length)), end="")
        print("{:5f}".format(i), end=" ")
    print()

    weights = train(weights, output)

    if output["output"][output["choice"]] >= 0.99:
        print('Converged on choice', output["choice"])
        break

print("Final weights\n", weights)