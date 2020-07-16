import k_means
import expectation_maximization as em
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import scipy.stats
import math
import array_to_latex as a2l


def visualize(gaussians, centroids, real_gaussians, clusters, scikit_clusters):
    fig, axs = plt.subplots(2)
    fig_1, axs_1 = plt.subplots(2)

    axs[0].set_title('k-means')
    axs[0].scatter([row[0] for row in gaussians[0]], [row[1] for row in gaussians[0]], color='green')
    axs[0].scatter([row[0] for row in gaussians[1]], [row[1] for row in gaussians[1]], color='red')
    axs[0].scatter([row[0] for row in gaussians[2]], [row[1] for row in gaussians[2]], color='blue')
    axs[0].scatter(centroids[0][0], centroids[0][1], color='black')
    axs[0].scatter(centroids[1][0], centroids[1][1], color='black')
    axs[0].scatter(centroids[2][0], centroids[2][1], color='black')

    axs[1].set_title('Groundtruth')
    axs[1].scatter(real_gaussians[0][0], real_gaussians[0][1], color='green')
    axs[1].scatter(real_gaussians[1][0], real_gaussians[1][1], color='red')
    axs[1].scatter(real_gaussians[2][0], real_gaussians[2][1], color='blue')

    axs_1[0].set_title('EM - custom')
    axs_1[0].scatter([row[0] for row in clusters[0]], [row[1] for row in clusters[0]], color='blue')
    axs_1[0].scatter([row[0] for row in clusters[1]], [row[1] for row in clusters[1]], color='red')
    axs_1[0].scatter([row[0] for row in clusters[2]], [row[1] for row in clusters[2]], color='green')

    axs_1[1].set_title('EM - SciKit-Learn')
    axs_1[1].scatter([row[0] for row in scikit_clusters[0]], [row[1] for row in scikit_clusters[0]], color='green')
    axs_1[1].scatter([row[0] for row in scikit_clusters[1]], [row[1] for row in scikit_clusters[1]], color='red')
    axs_1[1].scatter([row[0] for row in scikit_clusters[2]], [row[1] for row in scikit_clusters[2]], color='blue')

    fig.savefig('graph.png')
    fig_1.savefig('graph_1.png')
    fig.show()
    fig_1.show()


def sample_from_gaussian_distribution(mu, sigma, nbr_samples):
    c = np.zeros(100)
    z = np.zeros((100, 3))
    mean = [0, 0]
    cov = [[1, 0], [0, 100]]
    x, y = np.random.multivariate_normal(mu, sigma, 100).T

    return [x, y, c, z]


def main():
    number_of_samples = 100

    # create 100 samples each, from three different gaussian distributions
    gaussian_1 = sample_from_gaussian_distribution([-4, -2], [[2, 0], [0, 3]], number_of_samples)
    gaussian_2 = sample_from_gaussian_distribution([5.3, 4], [[1, 0], [0, 3]], number_of_samples)
    gaussian_3 = sample_from_gaussian_distribution([0, 1], [[3.5, 0], [0, 2]], number_of_samples)

    gaussians = [gaussian_1, gaussian_2, gaussian_3]

    centroid_clusters, centroids, real_gaussians = k_means.execute(gaussians)

    clusters = em.execute(gaussians)
    scikit_clusters = em.execute_with_scikit(gaussians)

    #Visualize
    visualize(centroid_clusters, centroids, real_gaussians, clusters, scikit_clusters)


if __name__ == '__main__':
    main()