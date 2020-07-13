import k_means
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import scipy.stats
import math
import array_to_latex as a2l

def visualize(gaussians, centroids):
    fig, axs = plt.subplots(1)

    axs.scatter([row[0] for row in gaussians[0]], [row[1] for row in gaussians[0]], color='green')
    axs.scatter([row[0] for row in gaussians[1]], [row[1] for row in gaussians[1]], color='red')
    axs.scatter([row[0] for row in gaussians[2]], [row[1] for row in gaussians[2]], color='blue')

    axs.scatter(centroids[0][0], centroids[0][1], color='green')
    axs.scatter(centroids[1][0], centroids[1][1], color='red')
    axs.scatter(centroids[2][0], centroids[2][1], color='blue')
    fig.savefig('graph.png')
    fig.show()


def sample_from_gaussian_distribution(x_min, x_max, mu, sigma, nbr_samples):
    x = np.linspace(x_min, x_max, nbr_samples)
    y = scipy.stats.norm.pdf(x, mu, sigma)
    c = np.zeros(nbr_samples)
    return [x, y, c]


def main():
    number_of_samples = 100

    # create 100 samples each, from three different gaussian distributions
    gaussian_1 = sample_from_gaussian_distribution(-6, -2, -4, 1.0, number_of_samples)
    gaussian_2 = sample_from_gaussian_distribution(-1, 3, 1, 1.9, number_of_samples)
    gaussian_3 = sample_from_gaussian_distribution(1.2, 5.2, 3.2, 0.6, number_of_samples)

    gaussians = [gaussian_1, gaussian_2, gaussian_3]

    centroid_clusters, centroids = k_means.main(gaussians)

    #Visualize
    visualize(centroid_clusters, centroids)


if __name__ == '__main__':
    main()