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

def create_random_points(number):
    points = np.ones((2, number))
    for i in range(number):
        x_i = random.uniform(0, 1)
        points[0][i] = x_i

    points[0] = selection_sort(points[0])

    for i in range(number):
        y_i = np.sin(2 * np.pi * points[0][i]) + random.uniform(-0.3, 0.3)
        points[1][i] = y_i

    return points


#stochastic gradient descend

def stochastic_gradient_descent(a, d, i, e, nbr_p, iteration):
    #parameters
    alpha = a
    degree = d
    interval = i
    epochs = e
    nbr_points = nbr_p

    # create random points
    points = create_random_points(nbr_points)

    #generate random parameters
    theta_j = np.ones(degree + 1)
    for i in range(len(theta_j)):
        theta_j[i] = random.uniform(-interval, interval)


    e_rms = np.zeros((2, epochs))
    #execute SGD
    for x in range(epochs):
        #iterate all data points
        mean_error = 0
        for i in range(len(points[0])):
            #solve polynom
            solve_polynom = 0
            for k in range(degree + 1):
                solve_polynom += theta_j[k] * points[0][i]**k

            #calculate error
            error = points[1][i] - solve_polynom
            mean_error += (solve_polynom - points[1][i]) ** 2
            for k in range(len(theta_j)):
                theta_j[k] = theta_j[k] + alpha * error * points[0][i]**k

        e_rms[0][x] = x
        e_rms[1][x] = np.sqrt(2 * (0.5 * mean_error) / nbr_points)

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

    #plot all
    fig, axs = plt.subplots(2)
    # fig.suptitle('Stochastic Gradient Descent Results')  # Iteration: ' + str(iteration))
    axs[0].scatter(points[0], points[1])
    axs[0].set_title('Data Points (blue scattered points), Estimation (red), Sinus Plot (green)')
    axs[0].set(xlabel='x-axis', ylabel='y-axis')
    axs[0].plot(sinus_plot[0], sinus_plot[1], color='green')
    axs[0].plot(points[0], final_plot, color='red')
    axs[1].plot(e_rms[0], e_rms[1])
    axs[1].set_title('RMS Error')
    axs[1].set(xlabel='Iterations', ylabel='Mean Squared Error')
    fig.show()

    error_differential = 0
    for i in range(len(e_rms[1])):
        error_differential += e_rms[1][i]

    final_parameters = ''
    for i in range(len(theta_j)):
        final_parameters += ' ' + str(theta_j[i])

    print(final_parameters)

    return error_differential
