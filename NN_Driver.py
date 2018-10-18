# This is NN_Driver.py

import numpy as np
import Iris_Data as iris
import NN_Lib as NN

# Changing this seed will affect dataset splitting as well as which seeds are generated for the weights
np.random.seed(10141996)

# Get training, validation, and test sets from iris data
training_data_arr, validation_data, test_data = iris.import_data()

# Initialize variable to store data on best model
overall_lowest_error = np.inf
overall_best_weights = []
best_seed = None

# Test different neural networks with same dimensions, random weights
for i in range(40):
    print("Network: ", i)

    # Change random seed to initialize weights and store for reproducibility
    seed = np.random.randint(1, 2147483647)
    np.random.seed(seed)

    # Set hyperparameters
    learning_rate = 0.01
    hidden_layer_size = 5

    # Create model
    model = NN.Model(hidden_layer_size, learning_rate)

    # Initialize tracking variables
    network_lowest_error = np.inf
    network_best_weights = [model.w_ih, model.w_ho]

    epoch = 1

    # Run through training until enough epochs have passed or error has increased 25%
    while epoch <= 10000 and model.average_error <= network_lowest_error * 1.25:
        if epoch % 100 == 0:
            print("Epoch: ", epoch)

        # Train in minibatches
        for data in training_data_arr:
            # Split answers from data
            l = np.hsplit(data, [-3])
            answers = l[1]
            data = l[0]

            # Forward Propagation
            model.forward_propagate(data)

            # Backward Propagation
            model.backward_propagate(answers)

            # Weight Updates
            model.update_weights()

        # Use cross-validation to determine how well the network performs
        # Split the data and answers
        l = np.hsplit(validation_data, [-3])
        answers = l[1]
        data = l[0]

        # Forward Propagation
        model.forward_propagate(data)

        # Calculate error
        model.calculate_error(answers)

        # If error from cross-validation is better than previous,
        # update lowest error and best weights with current value
        if model.average_error < network_lowest_error:
            network_lowest_error = model.average_error
            network_best_weights = [model.w_ih, model.w_ho][:]
            network_best_output = model.z_o

        if epoch % 100 == 0:
            print("Error: ", model.average_error)
        epoch += 1

    # If network is better than previous networks, update lowest error
    # and store best weights of network and seed used
    if network_lowest_error < overall_lowest_error:
        overall_lowest_error = network_lowest_error
        overall_best_weights = network_best_weights
        best_seed = seed
        print("Overall lowest error: ", overall_lowest_error)

# Test best network against test set to determine error on unseen data
model = NN.Model()
model.w_ih = overall_best_weights[0]
model.w_ho = overall_best_weights[1]

# Split data and answers
l = np.hsplit(test_data, [-3])
answers = l[1]
data = l[0]

# Forward Propagation
model.forward_propagate(data)

# Calculate cross entropy error
model.calculate_error(answers)

# Output results
print("Best seed: ", best_seed)
print("Average cross entropy error on test set: ", model.average_error)
print("Output on test set:\n", np.argmax(model.z_o, axis=1))
print("Answers of test set:\n", np.argmax(answers, axis=1))

for i in range(len(overall_best_weights)):
    np.savetxt("weights" + str(i) + ".txt", overall_best_weights[i])


# Output training sequence of best network for assignment

f = open("results.csv", "w")

# Use seed of best network
np.random.seed(best_seed)

# Set hyperparameters
learning_rate = 0.01
hidden_layer_size = 5

# Create model
model = NN.Model(hidden_layer_size, learning_rate)

# Initialize tracking variables
network_lowest_error = np.inf
network_best_weights = [model.w_ih, model.w_ho]

epoch = 1

# Run through training until enough epochs have passed or error has increased 25%
while epoch <= 10000 and model.average_error <= network_lowest_error * 1.25:

    output_text = "{},".format(epoch)

    training_errors = []

    # Train in minibatches
    for data in training_data_arr:
        # Split answers from data
        l = np.hsplit(data, [-3])
        answers = l[1]
        data = l[0]

        # Forward Propagation
        model.forward_propagate(data)

        # Backward Propagation
        model.backward_propagate(answers)

        # Calculate error
        model.calculate_error(answers)
        training_errors.append(model.average_error)

        # Weight Updates
        model.update_weights()

    # Add average of training set error to output text
    output_text += "{},".format(np.average(training_errors))

    # Use cross-validation to determine how well the network performs
    # Split the data and answers
    l = np.hsplit(validation_data, [-3])
    answers = l[1]
    data = l[0]

    # Forward Propagation
    model.forward_propagate(data)

    # Calculate error
    model.calculate_error(answers)

    # Output data to file
    output_text += "{}\n".format(model.average_error)
    f.write(output_text)

    # If error from cross-validation is better than previous,
    # update lowest error and best weights with current value
    if model.average_error < network_lowest_error:
        network_lowest_error = model.average_error
        network_best_weights = [model.w_ih, model.w_ho][:]
        network_best_output = model.z_o

    epoch += 1

f.close()
