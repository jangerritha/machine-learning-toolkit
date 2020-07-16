import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import scipy.stats
import scipy
import math
import array_to_latex as a2l
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



def transform_for_scikit(gaussians):
    x = []
    y = np.zeros(300)
    for j in range(3):
        for n in range(100):
            x.append([gaussians[j][0][n], gaussians[j][1][n]])


    for i in range(300):
        if i < 100:
            y[i] = 0
            continue
        if i < 200:
            y[i] = 1
            continue
        if i < 300:
            y[i] = 2
            continue

    return [x, y]


def calculate_distance(point_1, point_2):
    return np.sqrt((point_2[0] - point_1[0])**2 + (point_2[1] - point_1[1])**2)


def calculate_probability(x, y, phi, mu, cov, k):
    sum_p = 0
    p_k = 0

    for i in range(3):
        #print("Covariant: " + str(cov[i]))
        x_vector = np.array([x, y])
        mu_vector = np.array(mu[i])

        #print("x_vector: " + str(x_vector))
        #print("mu_vector: " + str(mu_vector))
        subtraction = x_vector - mu_vector
        #print("subtraction: " + str(subtraction))
        multiply_with_inverse = np.matmul(subtraction, np.linalg.inv(cov[i]))
        #print("multiply_with_inverse: " + str(multiply_with_inverse))
        multiply_with_subtraction = np.matmul(multiply_with_inverse, subtraction)
        #print("multiply_with_subtraction: " + str(multiply_with_subtraction))
        #print("base: " + str((1/(2*np.pi * np.linalg.det(cov[i])**0.5))))
        #print("exponential : " + str(np.exp(-0.5 * multiply_with_subtraction)))

        p_i = (1/(2*np.pi * np.sqrt(np.linalg.det(cov[i])))) * np.exp(-0.5 * multiply_with_subtraction)

        #print("PROBABILITY: " + str(p_i))

        if i == k:
            p_k = p_i

        sum_p += p_i * phi[i]

    p = (p_k * phi[k]) / sum_p

    #print(p)


    return p


def execute(gaussians):
    #initialize clusters with random values
    phi = np.zeros(3)
    for i in range(3):
        phi[i] = rnd.random()
    mu = np.ones((3, 2))
    cov = []
    for i in range(3):
        field = np.zeros((2, 2))
        field[0][0] = 8
        field[0][1] = 3
        field[1][0] = 7
        field[1][1] = 10
        cov.append(field)

    for n in range(100):
        for j in range(3):
            for k in range(3):
                gaussians[j][3][n][k] = rnd.random() * 0.01

    for i in range(1000):
        print("Iteration: " + str(i) + " -------------------------------------------------------------")
        for n in range(100):
            for j in range(3):
                for k in range(3):
                    probability = calculate_probability(gaussians[j][0][n], gaussians[j][1][n], phi, mu, cov, k)
                    #if not math.isnan(probability):
                    gaussians[j][3][n][k] = probability
                    #print(probability)

                #print("distribution" + str(gaussians[j][3][n]))

        print("Probability calculated -------------------------------------------------------------")

        for n in range(100):
            for j in range(3):
                 for k in range(3):
                     phi[k] += 1/300 * gaussians[j][3][n][k]


        #print("phi: " + str(phi))
        sum_w_x_0 = [0., 0., 0.]
        sum_w_x_1 = [0., 0., 0.]
        sum_w = [0., 0., 0.]

        for n in range(100):
            for j in range(3):
                 for k in range(3):
                    sum_w_x_0[k] += gaussians[j][3][n][k] * gaussians[j][0][n]
                    #print(gaussians[j][3][n][k])
                    sum_w_x_1[k] += gaussians[j][3][n][k] * gaussians[j][1][n]
                    sum_w[k] += gaussians[j][3][n][k]

        for u in range(3):
            mu[u][0] = sum_w_x_0[u] / sum_w[u]
            mu[u][1] = sum_w_x_1[u] / sum_w[u]

        #print("mu: " + str(mu))

        #print(mu)
        sum_cov_transpose = [0., 0., 0.]
        sum_cov = [0., 0., 0.]

        for n in range(100):
            for j in range(3):
                 for k in range(3):
                    subtraction = np.subtract([[gaussians[j][0][n], gaussians[j][1][n]]], [mu[k]])
                    #print([gaussians[j][0][n], gaussians[j][0][n]])
                    #print(mu[k])
                    multiply_transpose = np.dot(np.transpose(subtraction), subtraction)
                    sum_cov_transpose[k] += np.multiply(gaussians[j][3][n][k], multiply_transpose)
                    sum_cov[k] += gaussians[j][3][n][k]

        for u in range(3):
            cov[u] = sum_cov_transpose[u] / sum_cov[u]

        for n in range(100):
            for j in range(3):
                k = int(np.argmax(gaussians[j][3][n]))
                #print(k)
        #print("cov" + str(cov))

    cluster_0 = []
    cluster_1 = []
    cluster_2 = []

    clusters = [cluster_0, cluster_1, cluster_2]
    for n in range(100):
        for j in range(3):
            k = int(np.argmax(gaussians[j][3][n]))
            #print(k)
            clusters[k].append([gaussians[j][0][n], gaussians[j][1][n]])

    print("cov: " + str(cov))
    print("mu: " + str(mu))
    print("phi: " + str(phi))

    return clusters


def execute_with_scikit(gaussians):
    points = transform_for_scikit(gaussians)
    clf = LinearDiscriminantAnalysis()
    clf.fit(points[0], points[1])

    cluster_0 = []
    cluster_1 = []
    cluster_2 = []

    clusters = [cluster_0, cluster_1, cluster_2]
    for n in range(100):
        for j in range(3):
            k = clf.predict([[gaussians[j][0][n], gaussians[j][1][n]]])
            #print(int(k[0]))
            clusters[int(k[0])].append([gaussians[j][0][n], gaussians[j][1][n]])

    print(clf.get_params(deep=True))
    return clusters