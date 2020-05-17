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

    for n_p in range(0,2):
        temp_n_p = np.zeros((30,3))
        for i in range(30):
            temp_red  = np.zeros((24, 24))
            temp_green = np.zeros((24, 24))
            temp_blue = np.zeros((24, 24))

            for k in range(24):
                for n in range(24):
                    temp_red[k][n], temp_green[k][n], temp_blue[k][n] = data[n_p][i][k][n]

            temp_n_p[i][0] = min(temp_red.flatten())
            temp_n_p[i][1] = min(temp_green.flatten())
            temp_n_p[i][2] = min(temp_blue.flatten())

        ret.append(temp_n_p)

    return ret


def calculate_avg_values(data):
    ret = []

    for n_p in range(0,2):
        temp_n_p = np.zeros((30,3))
        for i in range(30):
            temp_red = np.zeros((24, 24))
            temp_green = np.zeros((24, 24))
            temp_blue = np.zeros((24, 24))

            for k in range(24):
                for n in range(24):
                    temp_red[k][n], temp_green[k][n], temp_blue[k][n] = data[n_p][i][k][n]

            temp_n_p[i][0] = sum(temp_red.flatten()) / len(temp_red.flatten())
            temp_n_p[i][1] = sum(temp_green.flatten()) / len(temp_green.flatten())
            temp_n_p[i][2] = sum(temp_blue.flatten()) / len(temp_blue.flatten())

        ret.append(temp_n_p)

    return ret


def calculate_average_for_col(col):
    col_1 = col_2 = col_3 = 0
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
    mu_1_mean[0:3] = min_mean_return
    avg_mean_return = calculate_average_for_col(avg_vals[1])
    mu_1_mean[3:6] = avg_mean_return

    #print(mu_1_mean)

    # calculate mu_0 for negative (0)
    mu_0_mean = np.zeros(6)

    # execute SGD
    #print(min_vals)
    min_mean_return = calculate_average_for_col(min_vals[0])
    #print(min_mean_return)
    mu_0_mean[0:3] = min_mean_return
    avg_mean_return = calculate_average_for_col(avg_vals[0])
    #print(avg_mean_return)
    mu_0_mean[3:6] = avg_mean_return
    #print(mu_0_mean)

    #print(mu_0_mean)


    #calculate temp vector
    tmp_vector_mu1 = np.zeros((30, 6))

    for i in range(30):
        tmp_vector_mu1[i][0] += min_vals[1][i][0] - mu_1_mean[0]
        tmp_vector_mu1[i][1] += min_vals[1][i][1] - mu_1_mean[1]
        tmp_vector_mu1[i][2] += min_vals[1][i][2] - mu_1_mean[2]
        tmp_vector_mu1[i][3] += avg_vals[1][i][0] - mu_1_mean[3]
        tmp_vector_mu1[i][4] += avg_vals[1][i][1] - mu_1_mean[4]
        tmp_vector_mu1[i][5] += avg_vals[1][i][2] - mu_1_mean[5]

    tmp_vector_mu0 = np.zeros((30, 6))

    for i in range(30):
        tmp_vector_mu0[i][0] += min_vals[0][i][0] - mu_0_mean[0]
        tmp_vector_mu0[i][1] += min_vals[0][i][1] - mu_0_mean[1]
        tmp_vector_mu0[i][2] += min_vals[0][i][2] - mu_0_mean[2]
        tmp_vector_mu0[i][3] += avg_vals[0][i][0] - mu_0_mean[3]
        tmp_vector_mu0[i][4] += avg_vals[0][i][1] - mu_0_mean[4]
        tmp_vector_mu0[i][5] += avg_vals[0][i][2] - mu_0_mean[5]

    #print(str(tmp_vector[0]))
    #print(str(tmp_vector[0].transpose(0)))


    #critical ----
    sigma_0 = 0
    sigma_1 = 0
    for i in range(30):
        sigma_1 += 1/m * np.outer(tmp_vector_mu1[i], tmp_vector_mu1[i].transpose())

    for i in range(30):
        sigma_0 += 1/m * np.outer(tmp_vector_mu0[i], tmp_vector_mu0[i].transpose())

    #print(str(sigma_0))
    #print(str(sigma_1))

    determinante_0 = np.linalg.det(sigma_0)
    determinante_1 = np.linalg.det(sigma_1)

    print(np.sqrt(determinante_0))
    print(np.sqrt(determinante_1))

    for i in range(30):
        x_val = np.zeros((1, 6))
        x_val[0][0] = min_vals[0][i][0]
        x_val[0][1] = min_vals[0][i][1]
        x_val[0][2] = min_vals[0][i][2]
        x_val[0][3] = avg_vals[0][i][0]
        x_val[0][4] = avg_vals[0][i][1]
        x_val[0][5] = avg_vals[0][i][2]

        probability_base_0 = 1/((2 * np.pi)**3 * np.sqrt(determinante_0))
        probability_base_1 = 1 / ((2 * np.pi) ** 3 * np.sqrt(determinante_1))

        #print(probability_base_0)
        #print(probability_base_1)

        x_min_mu0 = np.subtract(x_val, mu_0_mean)
        sigma0_invert = np.linalg.inv(sigma_0)
        transpose_mul_sigma0 = np.matmul(x_min_mu0.transpose(), x_min_mu0)

        p_y_0 = probability_base_0 * np.e** ((-1/2) * np.matmul(transpose_mul_sigma0, sigma0_invert))

        p_y_0 = np.sum(p_y_0)

        x_min_mu1 = np.subtract(x_val, mu_1_mean)
        sigma1_invert = np.linalg.inv(sigma_1)
        transpose_mul_sigma1 = np.matmul(x_min_mu1.transpose(), x_min_mu1)

        p_y_1 = probability_base_1 * np.e ** ((-1 / 2) * np.matmul(transpose_mul_sigma1, sigma1_invert))

        p_y_1 = np.sum(p_y_1)


        p_0 = (p_y_0 * 0.5)/(p_y_0 * 0.5) * (p_y_1 * 0.5)
        #print(p_0)

        p_1 = (p_y_1 * 0.5) / (p_y_0 * 0.5) * (p_y_1 * 0.5)
        #print(p_1)

        if p_0 > p_1:
            print('Image ' + str(i) + ': no Parasite' + ' P=0:' + str(p_0) + ' P=1:' + str(p_1))
        else:
            print('Image ' + str(i) + ': Parasite' + ' P=0:' + str(p_0) + ' P=1:' + str(p_1))

    for i in range(30):
        x_val = np.zeros((1, 6))
        x_val[0][0] = min_vals[1][i][0]
        x_val[0][1] = min_vals[1][i][1]
        x_val[0][2] = min_vals[1][i][2]
        x_val[0][3] = avg_vals[1][i][0]
        x_val[0][4] = avg_vals[1][i][1]
        x_val[0][5] = avg_vals[1][i][2]

        probability_base_0 = 1 / ((2 * np.pi) ** 3 * np.sqrt(determinante_0))
        probability_base_1 = 1 / ((2 * np.pi) ** 3 * np.sqrt(determinante_1))

        x_min_mu0 = np.subtract(x_val, mu_0_mean)
        sigma0_invert = np.linalg.inv(sigma_0)
        transpose_mul_sigma0 = np.matmul(x_min_mu0.transpose(), x_min_mu0)

        p_y_0 = probability_base_0 * np.e ** ((-1 / 2) * np.matmul(transpose_mul_sigma0, sigma0_invert))

        p_y_0 = np.sum(p_y_0)

        x_min_mu1 = np.subtract(x_val, mu_1_mean)
        sigma1_invert = np.linalg.inv(sigma_1)
        transpose_mul_sigma1 = np.matmul(x_min_mu1.transpose(), x_min_mu1)

        p_y_1 = probability_base_1 * np.e ** ((-1 / 2) * np.matmul(transpose_mul_sigma1, sigma1_invert))

        p_y_1 = np.sum(p_y_1)

        p_0 = (p_y_0 * 0.5) / (p_y_0 * 0.5) * (p_y_1 * 0.5)
        # print(p_0)

        p_1 = (p_y_1 * 0.5) / (p_y_0 * 0.5) * (p_y_1 * 0.5)
        # print(p_1)

        if p_0 > p_1:
            print('Image ' + str(i) + ': no Parasite' + ' P=0:' + str(p_0) + ' P=1:' + str(p_1))
        else:
            print('Image ' + str(i) + ': Parasite' + ' P=0:' + str(p_0) + ' P=1:' + str(p_1))

    #plot all
    #fig, axs = plt.subplots(1)
    #fig.savefig('graph.png')
    #fig.show()



if __name__ == '__main__':
    gaussian_discriminant()
