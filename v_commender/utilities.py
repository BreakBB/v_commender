import json
from random import shuffle


def save_recommendations_to_file(predicted_list):
    file_name = "predicted.json"

    # print("Saving " + str(len(predicted_list)) + " items to " + file_name, flush=True)
    with open(file_name, "w") as f:
        f.write(json.dumps(predicted_list))


def split_data_set(data_set):
    shuffle(data_set)
    split = int(len(data_set) * 0.8)
    training_set = data_set[:split]
    test_set = data_set[split:]

    return training_set, test_set
