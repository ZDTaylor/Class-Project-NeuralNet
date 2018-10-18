This project is a from-scratch implementation of a feedforward neural network with one hidden layer.  Data and a helper file for the Iris data set is provided, but NN_Lib and NN_Driver will support any dataset. If you use this code for a project, please cite me.

### Files:
- NN_Diagram.png - This is an illustration of the math required for forward and backward propagation when tracking data as matrices.
- NN Architecture.png - This is an illustration of the architecture implemented by NN_Driver.py
- IrisData.txt & IrisHelp.txt - This is the Iris dataset.
- Iris_Data.py - This imports the Iris dataset, encodes the class values, and splits the data into training, validation, and test data.  Each partition has an equal amount of each class.
- NN_Lib.py - This provides the Model class which implements the necessary methods to perform training. It also contains some helper functions for different activation and loss functions.
- NN_Driver.py - This creates and trains a group of neural networks on the Iris data and saves the results of training the best of those networks to results.csv 

NN_Driver.py can be run from an IDE or the command line and takes no arguments.
