import gradient_descent as sgd
import numpy as np
import random as rnd


def main():
    #initial parameters
    alpha = 0.001
    degree = 4
    interval = 0.5
    epochs = 6000
    nbr_points = 100

    optimize = False

    alpha_stepsize = 0.0005
    degree_stepsize = 1

    old_training_loss = 0
    for i in range(1):

        if i == 0:
            training_loss = sgd.stochastic_gradient_descent(alpha, degree, interval, epochs, nbr_points, i)
            old_training_loss = training_loss
            print('Iteration: ' + str(i) + ' Current Parameters: Degree: ' + str(degree) + ' Alpha: '
                  + str(alpha) + ' Total Loss: ' + str(training_loss))
            continue

        temp_alpha = 0
        temp_degree = 0

        if not optimize:
            temp_alpha = alpha
            temp_degree = degree

        if rnd.random() < 0.5 and optimize:
            temp_alpha = alpha + alpha_stepsize
        elif optimize:
            temp_alpha = alpha - alpha_stepsize
            if temp_alpha <= 0:
                temp_alpha += alpha_stepsize

        if rnd.random() < 0.5 and optimize:
            temp_degree = degree + degree_stepsize
        elif optimize:
            temp_degree = degree - degree_stepsize

        training_loss = sgd.stochastic_gradient_descent(temp_alpha, temp_degree, interval, epochs, nbr_points, i)

        accepted = False
        if training_loss < old_training_loss and optimize:
            alpha = temp_alpha
            degree = temp_degree
            accepted = True
            old_training_loss = training_loss

        print('Iteration: ' + str(i) + ' Current Parameters: Degree: ' + str(temp_degree) + ' Alpha: '
              + str(temp_alpha) + ' Total Loss: ' + str(training_loss) + ' Accepted: ' + str(accepted))


if __name__ == '__main__':
    main()

