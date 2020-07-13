import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import scipy.stats
import math
import array_to_latex as a2l


def calculate_distance(point_1, point_2):
    return np.sqrt((point_2[0] - point_1[0])**2 + (point_2[1] - point_1[1])**2)


def main(gaussians):
    #initialize centroids with random samples from gauss functions
    centroids = np.zeros((3, 2))
    for n in range(3):
        random_sample = rnd.randrange(100)
        #print(gaussians[n][0][random_sample])
        centroids[n][0] = gaussians[n][0][random_sample]
        centroids[n][1] = gaussians[n][1][random_sample]

    for i in range(5000):
        for n in range(100):
            for j in range(3):
                results = []
                for k in range(3):
                    results.append(calculate_distance(centroids[k], [gaussians[j][0][n], gaussians[j][1][n]]))

                gaussians[j][2][n] = int(np.argmin(results))
                #print(gaussians[j][2][n])

        centroid_1 = []
        centroid_2 = []
        centroid_3 = []
        centroid_clusters = [centroid_1, centroid_2, centroid_3]

        for n in range(100):
            for j in range(3):
                centroid_clusters[int(gaussians[j][2][n])].append([gaussians[j][0][n], gaussians[j][1][n]])

        for k in range(3):
            sum_up = np.sum(centroid_clusters[k], axis=0)
            centroids[k][0] = sum_up[0] / len(centroid_clusters[k])
            centroids[k][1] = sum_up[1] / len(centroid_clusters[k])

    return centroid_clusters, centroids



