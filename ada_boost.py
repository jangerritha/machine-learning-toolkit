import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import math
import array_to_latex as a2l


def read_points():
    data_f = open('data/assignment5/dataCircle.txt', 'r')
    data_list = data_f.readlines()
    points = []
    initial_weight = 1/len(data_list) #find Distribution for dataset, where sum is zero
    for i in range(len(data_list)):
        values = data_list[i].split()
        points.append([float(values[0]), float(values[1]), float(values[2]), initial_weight])

    return points


def visualize(positives, negatives, weak_classifiers):
    #-----plot_points-----
    positives_plot = np.zeros((2, 40))
    negatives_plot = np.zeros((2, 62))
    for (e, entry) in enumerate(positives):
        positives_plot[0][e] = entry[0]
        positives_plot[1][e] = entry[1]
    for (e, entry) in enumerate(negatives):
        negatives_plot[0][e] = entry[0]
        negatives_plot[1][e] = entry[1]

    fig, axs = plt.subplots(1)

    #------plot horizontal and vertical classifiers---------
    alpha = [c[3] for c in weak_classifiers]
    alpha_arg = np.argmax(alpha)
    normalize_alpha = 1/alpha[alpha_arg]
    for i in range(len(alpha)):
        alpha[i] *= normalize_alpha

    for (e, classifier) in enumerate(weak_classifiers):
        color = color_picker(alpha[e])
        if classifier[0]:
            plt.hlines(classifier[1], xmax=10.0, xmin=-10.0, colors=color)
        else:
            plt.vlines(classifier[1], ymax=10.0, ymin=-10.0, colors=color)


    #---plot all-----
    axs.scatter(positives_plot[0], positives_plot[1], color='green')
    axs.scatter(negatives_plot[0], negatives_plot[1], color='red')
    fig.savefig('graph.png')
    fig.show()
    latex_array = np.array(weak_classifiers)
    print(a2l.to_ltx(latex_array, frmt='{:6.2f}', arraytype='array'))


def color_picker(alpha):
    if alpha <= 0.1:
        return 'snow'
    if alpha <= 0.2:
        return 'white'
    if alpha <= 0.3:
        return 'whitesmoke'
    if alpha <= 0.4:
        return 'gainsboro'
    if alpha <= 0.5:
        return 'lightgray'
    if alpha <= 0.6:
        return 'silver'
    if alpha <= 0.7:
        return 'darkgray'
    if alpha <= 0.8:
        return 'grey'
    if alpha <= 0.9:
        return 'dimgray'
    if alpha <= 1.0:
        return 'black'


def validate_classifier(points, classifier):
    error_count = 0
    for pair in points:
        if classifier[0]:                  #horizontal line
            if pair[1] < classifier[1] and pair[2] == 1.0:  #y-coordinate lower than line and label true
                error_count += 1
            if pair[1] >= classifier[1] and pair[2] == -1.0:#y-coordinate higher than line and label false
                error_count += 1
        else:                               #vertical line
            if pair[0] < classifier[1] and pair[2] == 1.0: #x on left side of line and label true
                error_count += 1
            if pair[0] >= classifier[1] and pair[2] == -1.0: #x on right side of line and label false
                error_count += 1

    epsilon = 1/len(points) * error_count #calculate mean

    if epsilon > 0.5:  #flip if error > 0.5
        classifier[2] = True
        epsilon = 1 - epsilon

    if epsilon == 0.5:
        return None

    #count false classifications for each classifier
    return epsilon


def create_classifier():
    rnd_number = rnd.random()
    horizontal = True
    if rnd_number < 0.5:
        horizontal = False

    location = rnd.uniform(-10.0, 10.0) #Todo: step size
    #print(location)
    flipped = False
    return [horizontal, location, flipped, 0.0]


def predict_weak(x, y, classifier):
    result = 0
    if classifier[0]: #horizontal line
        if classifier[2] and y <= classifier[1]: #classifier was flipped
            result = 1
        elif not classifier[2] and y >= classifier[1]: #classifier was not flipped
            result = 1
        else:
            result = -1
    else:
        if classifier[2] and x <= classifier[1]:  # classifier was flipped
            result = 1
        elif not classifier[2] and x >= classifier[1]:  # classifier was not flipped
            result = 1
        else:
            result = -1
    #print(result)
    return result


def predict_strict(weak_classifiers, x, y):
    result_dirty = 0
    for classifier in weak_classifiers:
        alpha = classifier[3]
        result_dirty += alpha * predict_weak(x, y, classifier)
        #print(result_dirty)
        #print(classifier)
    return np.sign(result_dirty)


def compute_weighted_error_for_distribution(distibution, classifier, points):
    sigma = 0

    for (e, point) in enumerate(points):
        prediction = predict_weak(point[0], point[1], classifier)
        weight = distibution[e]
        if not prediction == point[2]:
            sigma += weight * 1

    return sigma


def update_distribution(distribution, points, alpha, classifier):
    temp_sum = 0
    for (e, point) in enumerate(points):
        #print(next_distribution)
        distribution[e] = distribution[e] * \
                               np.exp(-alpha * point[2] * predict_weak(point[0], point[1], classifier))
        temp_sum += distribution[e]

    for field in distribution:
        field *= (1/temp_sum)
    return distribution


def evaluate(points, weak_classifiers):
    correct_classified = 0
    for (e, point) in enumerate(points):
        result = predict_strict(weak_classifiers, point[0], point[1])
        #print(result)
        if result == point[2]:
            #if e % 5 == 0:
                #print('Correct: ' + str(result) + ' ' + str(point[2]))
            correct_classified += 1
        #else:
           # if e % 2 == 0:
                #print('False: ' + str(result) + ' ' + str(point[2]))

    print('Accuracy: ' + str(100 * correct_classified / len(points)) + '%, Total: ' + str(correct_classified) + ' of ' + str(len(points)))


def ada_boost():
    points = read_points()
    weak_classifiers = []
    distributions = []
    distributions.append([p[3] for p in points])
    distribution = distributions[0]

    for i in range(75):
        print('Iteration: ' + str(i))

        pool = []
        errors = []
        #Create classifier pool
        for n in range(100):
            classifier = create_classifier()
            err = validate_classifier(points, classifier)
            if err is None:
                while (err is None):
                    classifier = create_classifier()
                    err = validate_classifier(points, classifier)
            pool.append(classifier)

        for classifier in pool:
            epsilon = compute_weighted_error_for_distribution(distribution, classifier, points)
            errors.append(epsilon)

        error_array = np.asarray(errors)
        lowest_error_index = np.argmin(error_array)
        best_classifier = pool[lowest_error_index]
        best_epsilon = error_array[lowest_error_index]
        alpha = (1 / 2) * math.log((1 - best_epsilon) / best_epsilon)
        best_classifier[3] = alpha

        distribution = update_distribution(distribution, points, alpha, best_classifier)

        weak_classifiers.append(best_classifier)
        evaluate(points, weak_classifiers)

    evaluate(points, weak_classifiers)

    positives = points[0:40]
    negatives = points[41:102]
    visualize(positives, negatives, weak_classifiers)


if __name__ == '__main__':
    ada_boost()
