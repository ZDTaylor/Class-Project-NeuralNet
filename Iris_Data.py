import numpy as np


def import_data():
    flowers = {"Iris-setosa": [],
               "Iris-versicolor": [],
               "Iris-virginica": []}

    # Import Iris data
    with open("IrisData.txt", "r") as training_input:
        for line in training_input:
            line = line.rstrip("\n")
            line = line.split(",")
            flowers[line[-1]].append(list(map(float, line[:-1])))

    # Encode one-hot outputs for irises
    for i in range(50):
        flowers["Iris-setosa"][i] += [1, 0, 0]
        flowers["Iris-versicolor"][i] += [0, 1, 0]
        flowers["Iris-virginica"][i] += [0, 0, 1]

    # Ensure flower data is in random order
    np.random.shuffle(flowers["Iris-setosa"])
    np.random.shuffle(flowers["Iris-versicolor"])
    np.random.shuffle(flowers["Iris-virginica"])

    # Prepare training, validation, and test data in 60/20/20 split
    # Shuffle to ensure random training order and convert to numpy matrices
    training_data = []
    training_data += flowers["Iris-setosa"][:30]
    training_data += flowers["Iris-versicolor"][:30]
    training_data += flowers["Iris-virginica"][:30]
    np.random.shuffle(training_data)
    training_data = np.array(training_data)
    training_data_arr = np.split(training_data, 10)

    validation_data = []
    validation_data += flowers["Iris-setosa"][30:40]
    validation_data += flowers["Iris-versicolor"][30:40]
    validation_data += flowers["Iris-virginica"][30:40]
    np.random.shuffle(validation_data)
    validation_data = np.array(validation_data)

    test_data = []
    test_data += flowers["Iris-setosa"][40:]
    test_data += flowers["Iris-versicolor"][40:]
    test_data += flowers["Iris-virginica"][40:]
    np.random.shuffle(test_data)
    test_data = np.array(test_data)

    return training_data_arr, validation_data, test_data
