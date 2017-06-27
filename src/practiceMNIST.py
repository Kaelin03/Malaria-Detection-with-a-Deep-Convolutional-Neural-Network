#!/usr/bin/env python3

"""
Tutorial using a deep MLP to recognise handwritten digits.
From cambridgespark.com
"""

from keras.datasets import mnist                # Subroutines for getting MNIST data set
from keras.models import Model                  # Class for specifying and training a NN
from keras.layers import Input, Dense            # Types of NN layer to be used
from keras.utils import np_utils                # Utilities for one-hot encoding of ground truth values

"""
1. Define hyper-parameters
Hyper-parameters are assumed to be fixed before training starts
"""

batch_size = 128        # Number of training examples to consider at once in each iteration
num_epochs = 5         # Number of iterations over the entire training set
hidden_size = 512       # Number of neurons in each hidden layer

"""
2. Load and pre-process the MNIST data set
To pre-process input data:
    Flatten the images into 1D
    Threshold values to [0, 1]
Probabilistic classification requires  a single output neuron for each class
Need to transform the training output data into a "one-hot" encoding:
    For example, if the desired output class is 3, and there are 5 classes, 
    an appropriate one-hot encoding is [0, 0, 0, 1, 0]
"""

num_train = 60000       # Number of training examples
num_test = 10000        # Number of test examples

height = 28             # Image height
width = 28              # Image width
depth = 1               # Image is grayscale
num_classes = 10        # Number of distinct classes (1 per digit)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()    # Get MNIST data

X_train = X_train.reshape(num_train, height * width)        # Flatten data
X_test = X_test.reshape(num_test, height * width)           # Flatten data
X_train = X_train.astype("float32")                         # Convert to float 32
X_test = X_test.astype("float32")                           # Convert to float 32
X_train /= 255                                              # Normalise data in range [0, 1]
X_test /= 255                                               # Normalise data in range [0, 1]

Y_train = np_utils.to_categorical(Y_train, num_classes)     # One-hot encode the labels
Y_test = np_utils.to_categorical(Y_test, num_classes)      # One-hot encode the labels

"""
3. Define the model
We will use:
    Three Dense layers - fully unrestricted MLP structure
    (Linking all of the outputs of one layer to the inputs of the next layer)
    ReLU activations for the neurons in the first two layers
    Softmax activation in the final layer
NB: We only need to specify the size of the input layer
    Afterwards Keras will take care of initialising the weight variables with proper shapes
    Once all layers are defined, we need to identify the inputs and outputs
"""

inp = Input(shape=(height * width, ))                       # Our input is a 1D vector
hidden_1 = Dense(hidden_size, activation="relu")(inp)       # First hidden ReLu
hidden_2 = Dense(hidden_size, activation="relu")(hidden_1)  # Second hidden ReLU layer
out = Dense(num_classes, activation="softmax")(hidden_2)    # Output softmax layer
model = Model(inputs=inp, outputs=out)                      # Define the model

"""
4. Specify the loss function, optimisation algorithm and metrics to report
When dealing ith probabilistic classification, it is good to use the cross-entropy loss
This is because: 
    It only aims to maximise the models confidence in the correct class
    It is not concerned with the distribution of probabilities for other classes
    Squared-error loss would dedicate equal attention to minimising probabilities of other classes
The optimisation algorithm will typically revolve around some form of gradient descent
Key differences revolve around the manner in which the learning rate is chosen or adapted during training
Here, we will use the Adam optimiser
As our classes are balanced (in equal numbers), an appropriate metric to report is accuracy
Accuracy - the proportion of the inputs that are classified correctly
"""

model.compile(loss="categorical_crossentropy",              # Using the cross-entropy loss function
              optimizer="adam",                             # Using the Adam optimiser
              metrics=["accuracy"])                         # Report the accuracy

"""
5. Call the training algorithm
An excellent out-of-the-box feature of Keras is verbosity
    It is able to provide detailed real-time data of the training algorithm's progress
NB: validation_split =  Fraction of the training data to be used as validation data
                        The model will not train on this data
                        The model will evaluate the loss and anu model metrics on this data
    batch_size =        Number of samples per gradient update
    epochs =            The number of times to iterate over the training data arrays
"""

# Train the model using the training data
model.fit(X_train, Y_train, batch_size=batch_size,
          epochs=num_epochs,
          verbose=1,
          validation_split=0.1)

# Evaluate the model using the test data
scores = model.evaluate(X_test, Y_test, verbose=1)

print("\n")
for name, score in zip(* (model.metrics_names, scores)):
    print(name + ": " + str(round(score, 5)))



