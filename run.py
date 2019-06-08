import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_placeholders(n, m):
    X = tf.placeholder(tf.float32, shape=(m, None), name = 'X')
    Y = tf.placeholder(tf.float32, shape=(1, None), name = 'Y')
    return X, Y

def initialize_parameters():    
    W1 = tf.get_variable("W1", [25,62], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12,25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [12,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [1,12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [1,1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3 }
    
    return parameters

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.add(tf.matmul(W1, X) , b1)                            
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1) , b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2) , b3)
    return Z3

def compute_cost(predictions, Y):
	logits = tf.transpose(predictions)
	labels = tf.transpose(Y)
	cost = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(labels,logits=logits))

	return cost

def random_mini_batch(X, Y, size):
	indices = np.random.randint(0, X.shape[1], size)
	return X[:, indices], Y[:,indices].reshape((1, size))

def model(X_train, Y_train, X_test, Y_test, learning_rate = 1e-5,
          num_epochs = 100, minibatch_size = 128, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    (m, n) = X_train.shape                          # (n: num examples, m : num features)
    costs = []                                        # To keep track of the cost
    
    X, Y = create_placeholders(n, m)

    # Initialize parameters
    parameters = initialize_parameters()
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    predictions = forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(predictions, Y)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(n / minibatch_size) # number of minibatches of size minibatch_size in the train set
            for _ in range(num_minibatches):

                # Select a minibatch
                minibatch_X, minibatch_Y = random_mini_batch(X_train, Y_train, minibatch_size)
                
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        


        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        preds = tf.round(tf.sigmoid(forward_propagation(X, parameters)))
        correct_prediction = tf.equal(preds, Y)

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters


# execution starts here

def getInputData(filename):
	# returns tf array after processing csv file
	df = pd.read_csv(filename, index_col=False)
	df = df.drop(columns=['ID'])
	return np.asarray(df)

def getImprovementLabels(filename1, filename2):
	df1 = pd.read_csv(filename1, index_col=False)
	df2 = pd.read_csv(filename2, index_col=False)
	ID_to_rating_1 = dict(zip(df1['ID'], df1['Overall']))
	ID_to_rating_2 = dict(zip(df2['ID'], df2['Overall']))
	labels = [1 if ID_to_rating_2[player] > ID_to_rating_1[player] else 0 for player in df1['ID']]
	return np.asarray(labels)

X_train = getInputData('data/2017clean.csv')
Y_train = getImprovementLabels('data/2017clean.csv', 'data/2018clean.csv').reshape((1, -1))

X_test = getInputData('data/2018clean.csv')
Y_test = getImprovementLabels('data/2018clean.csv', 'data/2019clean.csv').reshape((1, -1))

n_x = X_test.shape[0]
dev_indices = np.random.randint(0, n_x, int(n_x * 0.5))

X_dev = X_test[dev_indices, :]
Y_dev = Y_test[:,dev_indices].reshape((1, -1))

parameters = model(X_train.T, Y_train, X_dev.T, Y_dev)
