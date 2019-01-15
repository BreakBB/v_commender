import json
import operator
import sys
import numpy as np

# Add the path so the script can be called from different directory and the module will be found
# This has to happen before any imports are made from "v_commender"
sys.path.append("../v_commender")

from v_commender.transform import movie_to_vector, create_training_data
from v_commender.utilities import save_recommendations_to_file, split_data_set
from v_commender.algorithms.neural_network import NeuralNetwork
from v_commender.algorithms.naive_bayes import NaiveBayesClassifier


def main():
    # Get the movie_list from the args and load it
    with open("data.json", "r") as f:
        script_data = json.loads(f.read())

    algo = sys.argv[1]
    voted_data = script_data['voted']

    if algo == "bayes":
        training_set, _ = split_data_set(voted_data)
        clf = train_bayes(training_set)
        recom_list = get_recommendations(clf, script_data['predict'])

        save_recommendations_to_file(recom_list)
    elif algo == "neural":
        training_set, _ = split_data_set(voted_data)
        nn = train_nn(training_set)
        recom_list = get_recommendations(nn, script_data['predict'])

        save_recommendations_to_file(recom_list)
    else:
        bayes_metrics = (0, 0, 0)
        nn_metrics = (0, 0, 0)

        rounds = 100

        for i in range(0, rounds):
            training_set, test_set = split_data_set(voted_data)

            clf = train_bayes(training_set)
            nn = train_nn(training_set)

            bayes_metrics = tuple(map(operator.add, bayes_metrics, get_metrics(clf, test_set, False)))
            nn_metrics = tuple(map(operator.add, nn_metrics, get_metrics(nn, test_set)))

        bayes_avrg = tuple(item_1 / rounds for item_1 in bayes_metrics)
        nn_avrg = tuple(item_1 / rounds for item_1 in nn_metrics)

        print("\nBayes:")
        print("Precision: " + str(bayes_avrg[0]))
        print("Recall: " + str(bayes_avrg[1]))
        print("F-Measure: " + str(bayes_avrg[2]))

        print("\nNeural Network:")
        print("Precision: " + str(nn_avrg[0]))
        print("Recall: " + str(nn_avrg[1]))
        print("F-Measure: " + str(nn_avrg[2]))
    return


def train_bayes(training_set):
    training_data = create_training_data(training_set)

    clf = NaiveBayesClassifier()
    clf.train(training_data['movies'], np.ravel(training_data['votes']))

    return clf


def train_nn(training_set):
    training_data = create_training_data(training_set, True)

    nn = NeuralNetwork(training_data['movies'], training_data['votes'])

    # train the NN 1,500 times
    for i in range(1500):
        # print("Loss: \n" + str(nn.get_loss()) + "\n", flush=True)
        nn.train()

    return nn


def get_recommendations(algo, movies_to_predict):
    recom_list = []

    for movie in movies_to_predict:
        predict_data = movie_to_vector(movie, True)
        movie_id = movie['movie_id']
        predicted = algo.predict(predict_data)
        recom_list.append({
            "id": movie_id,
            "value": predicted[0]
        })

        # i = 0
        # for movie in script_data['predict']:
        #     movie_id = movie['movie_id']
        #     predicted_list.append({
        #         "id": movie_id,
        #         "value": predicted[i]
        #     })
        #     print("Predict for '" + movie_id + "': " + str(predicted[i]), flush=True)
        #     i += 1

        # print("Predict for '" + movie_id + "': " + str(predicted), flush=True)
    return recom_list


def get_metrics(algo, test_set, norm=True):
    true_positives = 0
    false_negatives = 0
    false_positives = 0

    for movie in test_set:
        vector = movie_to_vector(movie, norm)
        predicted = algo.predict(vector)[0]

        positive_prediction = predicted > 0.7
        negative_prediction = predicted <= 0.7

        if movie['vote'] and positive_prediction:
            true_positives += 1
        elif movie['vote'] and negative_prediction:
            false_negatives += 1
        elif not movie['vote'] and positive_prediction:
            false_positives += 1

    precision = 0
    recall = 0
    f_measure = 0

    if (true_positives + false_positives) > 0:
        precision = true_positives / (true_positives + false_positives)  # How useful

    if (true_positives + false_negatives) > 0:
        recall = true_positives / (true_positives + false_negatives)  # How complete

    if (precision + recall) > 0:
        f_measure = 2 * (precision * recall / (precision + recall))

    # print("Precision: " + str(precision))
    # print("Recall: " + str(recall))
    # print("F-Measure: " + str(f_measure))

    return precision, recall, f_measure


if __name__ == '__main__':
    main()
