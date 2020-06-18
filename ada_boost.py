import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import math


def read_points():
    data_f = open('data/assignment5/dataCircle.txt', 'r')
    data_list = data_f.readlines()
    points = []
    initial_weight = 1/len(data_list) #find Distribution for dataset, where sum is zero
    for i in range(len(data_list)):
        values = data_list[i].split()
        points.append([float(values[0]), float(values[1]), float(values[2]), initial_weight])

    return points


def visualize(positives, negatives):
    positives_plot = np.zeros((2, 40))
    negatives_plot = np.zeros((2, 62))
    for (e, entry) in enumerate(positives):
        positives_plot[0][e] = entry[0]
        positives_plot[1][e] = entry[1]
    for (e, entry) in enumerate(negatives):
        negatives_plot[0][e] = entry[0]
        negatives_plot[1][e] = entry[1]
    # plot all
    fig, axs = plt.subplots(1)
    axs.scatter(positives_plot[0], positives_plot[1], color='green')
    axs.scatter(negatives_plot[0], negatives_plot[1], color='red')
    fig.savefig('graph.png')
    fig.show()


def validate_classifier(points, classifier):
    error_count = 0
    for pair in points:
        if classifier[0]:
            if pair[1] < classifier[1] and pair[2] == 1.0:
                error_count += 1
            if pair[1] >= classifier[1] and pair[2] == -1.0:
                error_count += 1
        else:
            if pair[0] < classifier[1] and pair[2] == 1.0:
                error_count += 1
            if pair[0] >= classifier[1] and pair[2] == -1.0:
                error_count += 1

    epsilon = 1/len(points) * error_count

    if epsilon > 0.5:
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

    location = rnd.uniform(-10.0, 10.0)
    flipped = False
    return [horizontal, location, flipped, 0.0]


def predict_weak(x, y, classifier):
    result = 0
    if classifier[0]: #horizontal line
        if classifier[2] and y <= classifier[1]: #classifier was flipped
            result = 1
        elif y >= classifier[1]: #classifier was not flipped
            result = 1
        else:
            result = -1
    else:
        if classifier[2] and x <= classifier[1]:  # classifier was flipped
            result = 1
        elif x >= classifier[1]:  # classifier was not flipped
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
    return np.sign(result_dirty)


def compute_weighted_error_for_distribution(distibution, classifier, points):
    sigma = 0
    for (e, point) in enumerate(points):
        prediction = predict_weak(point[0], point[1], classifier)
        weight = distibution[e]
        if not prediction == point[2]:
            #print('bullet0')
            sigma += weight * 1

    return sigma


def train(distributions, weak_classifiers, points):
    #Calculate epsilon for every weak classifier
    error_per_classifier = []
    for (e, classifier) in enumerate(weak_classifiers):
        distribution = distributions[e]
        weighted_epsilon = compute_weighted_error_for_distribution(distribution, classifier, points)
        error_per_classifier.append(weighted_epsilon)

    #find best weak classifier
    best_epsilon_arg = np.argmin(error_per_classifier)
    best_epsilon = error_per_classifier[best_epsilon_arg]

    #calculate alpha
    alpha = (1 / 2) * math.log((1 - best_epsilon) / best_epsilon)
    print('Lowest Epsilon: ' + str(best_epsilon) + ', Best Alpha: ' + str(alpha))

    #Set Alpha to make strong hypothesis
    weak_classifiers[best_epsilon_arg][3] = alpha

    #Create next distribution D_t+1
    next_distribution = update_distribution(distributions[best_epsilon_arg]
                                            , points, alpha, best_epsilon, weak_classifiers[best_epsilon_arg])
    distributions.append(next_distribution)


def update_distribution(distribution, points, alpha, epsilon, classifier):
    next_distribution = distribution
    for (e, point) in enumerate(points):
        #print(next_distribution)
        z = 1 / (2 * np.sqrt(epsilon * (1 - epsilon)))
        next_distribution[e] = \
            z * next_distribution[e] * \
            np.exp(-alpha * point[2] * predict_weak(point[0], point[1], classifier))
    return next_distribution


def evaluate(points, weak_classifiers):
    correct_classified = 0
    for point in points:
        result = predict_strict(weak_classifiers, point[0], point[1])
        if result == point[2]:
            correct_classified += 1

    print('Accuracy: ' + str(correct_classified / len(points)) + '%, Total: ' + str(correct_classified) + ' of ' + str(len(points)))


def ada_boost():
    points = read_points()
    weak_classifiers = []
    distributions = []
    distributions.append([p[3] for p in points])
    #distribution = distribution[0]

    #print(points)
    #print(points)
    for i in range(500):
        print('Iteration: ' + str(i))

        #Create classifier
        classifier = create_classifier()

        #calculate Epsilon
        epsilon = validate_classifier(points, classifier)
        if epsilon is None:
            while(epsilon is None):
                classifier = create_classifier()
                epsilon = validate_classifier(points, classifier)

        weak_classifiers.append(classifier)

        train(distributions, weak_classifiers, points)
        print(weak_classifiers)
        print(distributions)
        evaluate(points, weak_classifiers)

    #print('bullet1: ' + str(weak_classifiers))
    evaluate(points, weak_classifiers)

    positives = points[0:40]
    negatives = points[41:102]
    visualize(positives, negatives)


if __name__ == '__main__':
    ada_boost()
