"""

"""

import numpy as np, random, math

print('Testing weighted features')

LEARNING_RATE = 0.0001
FEATURES = 2
REWARDS = [1,2,3,3.1]
OUTPUTS = len(REWARDS)

weights = np.array([[random.random() for i in range(FEATURES)] for j in range(OUTPUTS)])
weights = np.array([
    np.array([0.0,4.0]),
    np.array([0.0,3.0]),
    np.array([0.0,2.0]),
    np.array([0.0,1.0])
])

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
    for i in range(len(grad)):
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

    if output["output"][output["choice"]] >= 0.98:
        print('Converged on choice', output["choice"])
        break

print("Final weights\n", weights)