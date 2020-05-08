import numpy as np
import random
import matplotlib.pyplot as plt


#this selection sort implementation was copied from https://jakevdp.github.io/PythonDataScienceHandbook/02.08-sorting.html
def selection_sort(x):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return x
#--------------------------------------------------


def read_points():
    data_f = open('data/assignment2/data.txt', 'r')
    data_list = data_f.readlines()
    points = np.ones((3, len(data_list)))
    for i in range(len(data_list)):
        values = data_list[i].split()
        points[0][i] = values[0]
        points[1][i] = values[1]
        points[2][i] = values[2]

    return points


#stochastic gradient descend

def stochastic_gradient_descent():#a, d, i, e, iteration):
    #parameters
    alpha = 0.001
    degree = 2 #fixed
    interval = 0.01
    epochs = 6000

    # create random points
    points = read_points()

    #generate random parameters
    theta_j = np.ones(degree + 1)
    for i in range(len(theta_j)):
        theta_j[i] = random.uniform(-interval, interval)


    #e_rms = np.zeros((2, epochs))
    #execute SGD
    for x in range(epochs):
        #iterate all data points
        mean_error = 0
        for i in range(len(points[0])):
            #solve
            solve = theta_j[0] + theta_j[1] * points[0][i] + theta_j[2] * points[1][i]

            #calculate error
            error = points[2][i] - 1/(1 + np.e ** (-solve))
            #print('Iteration: ' + str(x) + ' ' + str(error))
            mean_error += error
            point_s = np.zeros(2)
            theta_j[0] = theta_j[0] + alpha * error * 1.0
            theta_j[1] = theta_j[1] + alpha * error * points[0][i]
            theta_j[2] = theta_j[2] + alpha * error * points[1][i]

            #print(str(theta_j))

        #e_rms[0][x] = x
        print(str(mean_error / len(points[0])))

    #calculate approximation graph for plot
    final_plot = np.zeros(len(points[0]))
    for i in range(len(points[0])):
        solve_polynom = 0
        for k in range(degree + 1):
            solve_polynom += theta_j[k] * points[0][i] ** k

        final_plot[i] = solve_polynom

    sinus_plot = np.zeros((2, 100))

    for i in range(len(sinus_plot[0])):
        sinus_plot[0][i] = 0.01 * i
        sinus_plot[1][i] = np.sin(2 * np.pi * sinus_plot[0][i])

    blue_points = np.zeros((2, 50))
    green_points = np.zeros((2, 50))

    for i in range(len(points[0])):
        if i < 50:
            blue_points[0][i] = points[0][i]
            blue_points[1][i] = points[1][i]
        else:
            x = i
            x -= 50
            green_points[0][x] = points[0][i]
            green_points[1][x] = points[1][i]

    final_plot = np.zeros((2, len(points[0])))
    #temp_points = selection_sort(points[0])
    for i in range(len(points[0])):
        solve = 0
        solve = (theta_j[0] + theta_j[1] * points[0][i]) * (-1/theta_j[2])

        #final_plot[0][i] = points[0][i]
        #final_plot[1][i] = solve

    #plot all
    fig, axs = plt.subplots(1)
    axs.scatter(green_points[0], green_points[1], color='green')
    axs.scatter(blue_points[0], blue_points[1], color='blue')
    axs.plot(final_plot[0], final_plot[1], color="black")
    fig.show()

    #error_differential = 0
    #for i in range(len(e_rms[1])):
        #error_differential += e_rms[1][i]

    #final_parameters = ''
    #for i in range(len(theta_j)):
        #final_parameters += ' ' + str(theta_j[i])

    #print(final_parameters)

    #return error_differential


if __name__ == '__main__':
    stochastic_gradient_descent()
