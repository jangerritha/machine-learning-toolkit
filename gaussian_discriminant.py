import numpy as np
import random
import matplotlib.pyplot as plt
from imageio import imread


#this selection sort implementation was copied from https://jakevdp.github.io/PythonDataScienceHandbook/02.08-sorting.html
def selection_sort(x):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return x
#--------------------------------------------------


def create_data():
    negatives = []
    positives = []

    for i in range(30):
        path = 'data/assignment3/negatives/n'
        if i < 9:
            path += '0'
        negatives.append(imread(path + str(i + 1) + '.png'))

    for i in range(30):
        path = 'data/assignment3/positives/p'
        if i < 9:
            path += '0'
        positives.append(imread(path + str(i + 1) + '.png'))

    return [negatives, positives]


def calculate_min_values(data):
    negatives = data[0]
    positives = data[1]

    n_min_vals = np.zeros((30, 3))

    for i in range(30):
        temp_red = np.zeros((24, 24))
        temp_green = np.zeros((24, 24))
        temp_blue = np.zeros((24, 24))

        for k in range(24):
            for n in range(24):
                temp_red[k][n] = negatives[i][k][n][0]
                temp_green[k][n] = negatives[i][k][n][1]
                temp_blue[k][n] = negatives[i][k][n][2]

        n_min_vals[i][0] = min(temp_red.flatten())
        n_min_vals[i][1] = min(temp_green.flatten())
        n_min_vals[i][2] = min(temp_blue.flatten())

    p_min_vals = np.zeros((30, 3))

    for i in range(30):
        temp_red = np.zeros((24, 24))
        temp_green = np.zeros((24, 24))
        temp_blue = np.zeros((24, 24))

        for k in range(24):
            for n in range(24):
                temp_red[k][n] = positives[i][k][n][0]
                temp_green[k][n] = positives[i][k][n][1]
                temp_blue[k][n] = positives[i][k][n][2]

        p_min_vals[i][0] = min(temp_red.flatten())
        p_min_vals[i][1] = min(temp_green.flatten())
        p_min_vals[i][2] = min(temp_blue.flatten())

    return [n_min_vals, p_min_vals]


def calculate_avg_values(data):
    negatives = data[0]
    positives = data[1]

    n_avg_vals = np.zeros((30, 3))

    for i in range(30):
        temp_red = np.zeros((24, 24))
        temp_green = np.zeros((24, 24))
        temp_blue = np.zeros((24, 24))

        for k in range(24):
            for n in range(24):
                temp_red[k][n] = negatives[i][k][n][0]
                temp_green[k][n] = negatives[i][k][n][1]
                temp_blue[k][n] = negatives[i][k][n][2]

        n_avg_vals[i][0] = sum(temp_red.flatten()) / len(temp_red.flatten())
        n_avg_vals[i][1] = sum(temp_green.flatten()) / len(temp_green.flatten())
        n_avg_vals[i][2] = sum(temp_blue.flatten()) / len(temp_blue.flatten())

    p_avg_vals = np.zeros((30, 3))

    for i in range(30):
        temp_red = np.zeros((24, 24))
        temp_green = np.zeros((24, 24))
        temp_blue = np.zeros((24, 24))

        for k in range(24):
            for n in range(24):
                temp_red[k][n] = positives[i][k][n][0]
                temp_green[k][n] = positives[i][k][n][1]
                temp_blue[k][n] = positives[i][k][n][2]

        p_avg_vals[i][0] = sum(temp_red.flatten()) / len(temp_red.flatten())
        p_avg_vals[i][1] = sum(temp_green.flatten()) / len(temp_green.flatten())
        p_avg_vals[i][2] = sum(temp_blue.flatten()) / len(temp_blue.flatten())

    return [n_avg_vals, p_avg_vals]


def calculate_average_for_col(col):
    col_1 = 0
    col_2 = 0
    col_3 = 0
    for i in range(len(col[0])):
        col_1 += col[0][i]
        col_2 += col[1][i]
        col_3 += col[2][i]

    col_1 = col_1 / len(col[0])
    col_2 = col_2 / len(col[0])
    col_3 = col_3 / len(col[0])

    return [col_1, col_2, col_3]


def gaussian_discriminant():
    #parameters
    alpha = 0.1
    degree = 2 #fixed
    interval = 0.01
    epochs = 6000

    # create data as 24x24x3 matrixes from images
    data = create_data()

    #calculate min rgb values for every image
    min_vals = calculate_min_values(data)

    # calculate average rgb values for every image
    avg_vals = calculate_avg_values(data)

    # ----------------- calculate parameters ----------------------
    #calculate phi
    m = 60
    phi = 30/m

    #calculate mu_1 for positive (1)
    mu_1_mean = np.zeros(6)

    #execute SGD
    min_mean_return = calculate_average_for_col(min_vals[1])
    mu_1_mean[0] = min_mean_return[0]
    mu_1_mean[1] = min_mean_return[1]
    mu_1_mean[2] = min_mean_return[2]
    avg_mean_return = calculate_average_for_col(avg_vals[1])
    mu_1_mean[3] = min_mean_return[0]
    mu_1_mean[4] = min_mean_return[1]
    mu_1_mean[5] = min_mean_return[2]

    #print(mu_1_mean)

    # calculate mu_0 for negative (0)
    mu_0_mean = np.zeros(6)

    # execute SGD
    min_mean_return = calculate_average_for_col(min_vals[0])
    mu_0_mean[0] = min_mean_return[0]
    mu_0_mean[1] = min_mean_return[1]
    mu_0_mean[2] = min_mean_return[2]
    avg_mean_return = calculate_average_for_col(avg_vals[0])
    mu_0_mean[3] = min_mean_return[0]
    mu_0_mean[4] = min_mean_return[1]
    mu_0_mean[5] = min_mean_return[2]

    #print(mu_0_mean)


    #calculate temp vector
    tmp_vector = np.zeros((60, 6))

    for i in range(30):
        tmp_vector[i][0] += min_vals[1][i][0] - mu_1_mean[0]
        tmp_vector[i][1] += min_vals[1][i][1] - mu_1_mean[1]
        tmp_vector[i][2] += min_vals[1][i][2] - mu_1_mean[2]
        tmp_vector[i][3] += avg_vals[1][i][0] - mu_1_mean[3]
        tmp_vector[i][4] += avg_vals[1][i][1] - mu_1_mean[4]
        tmp_vector[i][5] += avg_vals[1][i][2] - mu_1_mean[5]

    for i in range(30):
        tmp_vector[i+30][0] += min_vals[0][i][0] - mu_0_mean[0]
        tmp_vector[i+30][1] += min_vals[0][i][1] - mu_0_mean[1]
        tmp_vector[i+30][2] += min_vals[0][i][2] - mu_0_mean[2]
        tmp_vector[i+30][3] += avg_vals[0][i][0] - mu_0_mean[3]
        tmp_vector[i+30][4] += avg_vals[0][i][1] - mu_0_mean[4]
        tmp_vector[i+30][5] += avg_vals[0][i][2] - mu_0_mean[5]

    sigma = np.zeros(6)

    for x in range(6):
        temp_val = 0
        for i in range(60):
            temp_val += tmp_vector[i][x]*tmp_vector


    print(str(tmp_vector))
    #calculate approximation graph for plot

    #plot all
    fig, axs = plt.subplots(1)
    fig.savefig('graph.png')
    fig.show()



if __name__ == '__main__':
    gaussian_discriminant()
