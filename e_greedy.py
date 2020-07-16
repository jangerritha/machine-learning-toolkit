import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import scipy.stats
import math
import array_to_latex as a2l


def bandit(Q, a):
    return np.random.normal(loc=Q[a])


def select_action(Q, epsilon):
    random_number = np.random.normal()
    if random_number < epsilon:
        return rnd.randrange(10)
    else:
        return np.argmax(Q)


def e_greedy():
    epsilon = 0.0
    Q = np.zeros((10))
    N = np.zeros((10))

    for i in range(1000):
        a = select_action(Q, epsilon)
        reward = bandit(Q, a)
        N[a] += 1
        Q[a] += 1 / N[a] * (reward - Q[a])

    return [Q, N]


def main():
    global_Q = np.zeros((10))
    global_N = np.zeros((10))
    for n in range(2000):
        print("Iteration: " + str(n) + " of 2000")
        q, n = e_greedy()
        for k in range(10):
            global_Q[k] += 1/2000 * q[k]
            global_N[k] += 1 / 2000 * n[k]

    print(global_Q)
    print(global_N)


if __name__ == '__main__':
    main()