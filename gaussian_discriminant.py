import numpy as np
import random
import matplotlib.pyplot as plt
from imageio import imread


def create_data():
    ret = []
    strn_p = [ "negatives","positives",'n','p']

    for n_p in range(0,2):
        temp_n_p = []
        for i in range(30):
            path = "data/assignment3/"+strn_p[n_p]+'/'+strn_p[n_p+2]
            if i < 9:
                path += '0'
            temp_n_p.append(imread(path + str(i + 1) + '.png'))
        ret.append(temp_n_p)
    return ret


def calculate_min_values(data):
    ret = []
    temp_red = temp_green = temp_blue = np.zeros((24, 24))

    for n_p in range(0,2):
        temp_n_p = np.zeros((30,3))
        for i in range(30):
            for k in range(24):
                for n in range(24):
                    temp_red[k][n], temp_green[k][n], temp_blue[k][n] = data[n_p][i][k][n]

            temp_n_p[i] = [min(temp_red.flatten()), min(temp_green.flatten()), min(temp_blue.flatten())]

        ret.append(temp_n_p)

    return ret

def calculate_max_values(data):
    ret = []
    temp_red = temp_green = temp_blue = np.zeros((24, 24))

    for n_p in range(0,2):
        temp_n_p = np.zeros((30,3))
        for i in range(30):
            for k in range(24):
                for n in range(24):
                    temp_red[k][n], temp_green[k][n], temp_blue[k][n] = data[n_p][i][k][n]

            temp_n_p[i] = [max(temp_red.flatten()), max(temp_green.flatten()), max(temp_blue.flatten())]

        ret.append(temp_n_p)

    return ret

def calculate_avg_values(data):
    ret = []
    temp_red = temp_green = temp_blue = np.zeros((24, 24))

    for n_p in range(0,2):
        temp_n_p = np.zeros((30,3))
        for i in range(30):
            for k in range(24):
                for n in range(24):
                    temp_red[k][n], temp_green[k][n], temp_blue[k][n] = data[n_p][i][k][n]

            temp_n_p[i] = [ sum(temp_red.flatten()) / len(temp_red.flatten()),
                    sum(temp_green.flatten()) / len(temp_green.flatten()),
                    sum(temp_blue.flatten()) / len(temp_blue.flatten())]

        ret.append(temp_n_p)

    return ret


def calculate_average_for_col(col):
    col_1 = col[0].sum(axis=0) / len(col[0])
    col_2 = col[1].sum(axis=0) / len(col[0])
    col_3 = col[2].sum(axis=0) / len(col[0])

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

    max_vals = calculate_max_values(data)

    # ----------------- calculate parameters ----------------------
    #calculate phi
    m = 60
    phi = 30/m

    #calculate mu_1 for positive (1)
    mu_1_mean = np.zeros(9)

    #execute SGD
    min_mean_return = calculate_average_for_col(min_vals[1])
    mu_1_mean[0:3] = min_mean_return
    avg_mean_return = calculate_average_for_col(avg_vals[1])
    mu_1_mean[3:6] = avg_mean_return
    max_mean_return = calculate_average_for_col(avg_vals[1])
    mu_1_mean[6:9] = max_mean_return

    # calculate mu_0 for negative (0)
    mu_0_mean = np.zeros(9)

    # execute SGD
    min_mean_return = calculate_average_for_col(min_vals[0])
    mu_0_mean[0:3] = min_mean_return
    avg_mean_return = calculate_average_for_col(avg_vals[0])
    mu_0_mean[3:6] = avg_mean_return
    max_mean_return = calculate_average_for_col(avg_vals[0])
    mu_0_mean[6:9] = max_mean_return

    #calculate temp vector
    tmp_vector_mu1 = []
    tmp_vector_mu0 = []

    for i in range(30):
        f_0, f_1, f_2 = min_vals[1][i] - mu_1_mean[0:3]
        f_3, f_4, f_5 = avg_vals[1][i] - mu_1_mean[3:6]
        f_6, f_7, f_8 = max_vals[1][i] - mu_1_mean[6:9]

        tmp_vector_mu1.append([f_0, f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8])

        f_0, f_1, f_2 = min_vals[0][i] - mu_0_mean[0:3]
        f_3, f_4, f_5 = avg_vals[0][i] - mu_0_mean[3:6]
        f_6, f_7, f_8 = max_vals[0][i] - mu_0_mean[6:9]
        tmp_vector_mu0.append([f_0, f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8])


    sigma = 0
    for i in range(60):
        if i < 30:
            sigma += (1/m) * np.matmul(np.reshape(tmp_vector_mu1[i], (1, 9)).transpose(), np.reshape(tmp_vector_mu1[i], (1, 9)))
        else:
            sigma += (1/m) * np.matmul(np.reshape(tmp_vector_mu0[i - 30], (1, 9)).transpose(), np.reshape(tmp_vector_mu0[i - 30], (1, 9)))

    determinante = np.linalg.det(sigma)

    strP_nP = ["Parasite", "no Parasite"]
    for i in range(60):
        x_val = np.zeros((1, 9))
        if i < 30:
            x_val[0][0:3] = min_vals[0][i]
            x_val[0][3:6] = avg_vals[0][i]
            x_val[0][6:9] = max_vals[0][i]
        else:
            x_val[0][0:3] = min_vals[1][i - 30]
            x_val[0][3:6] = avg_vals[1][i - 30]
            x_val[0][6:9] = max_vals[1][i - 30]


        probability_base = 1 / ((2 * np.pi) ** 3 * np.sqrt(determinante))

        x_min_mu0 = np.subtract(x_val, mu_0_mean)
        sigma0_invert = np.linalg.matrix_power(sigma, -1)
        transpose_mul_sigma0 = np.matmul(x_min_mu0.transpose(), x_min_mu0)
        p_y_0 = probability_base * np.exp((-1) * (1 / 2) * np.matmul(transpose_mul_sigma0, sigma0_invert))

        x_min_mu1 = np.subtract(x_val, mu_1_mean)
        sigma1_invert = np.linalg.matrix_power(sigma, -1)
        transpose_mul_sigma1 = np.matmul(x_min_mu1.transpose(), x_min_mu1)
        p_y_1 = probability_base * np.exp((-1) * (1 / 2) * np.matmul(transpose_mul_sigma1, sigma1_invert))

        p_y_add = np.add((p_y_0 * 0.5), (p_y_1 * 0.5))

        p_0 = (p_y_0 * 0.5) / p_y_add
        p_0 = p_0.flatten()[np.argmax(p_0.flatten())]

        p_1 = (p_y_1 * 0.5) / p_y_add
        p_1 = p_1.flatten()[np.argmax(p_1.flatten())]


        switch_pn = 0
        if p_0 > p_1:
            switch_pn = 1

        print('Image ' + str(i) + ':\t' + strP_nP[switch_pn] + '\tP=0:' + str(p_0) + '\tP=1:' + str(p_1))


if __name__ == '__main__':
    gaussian_discriminant()
