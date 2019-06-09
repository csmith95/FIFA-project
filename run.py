import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_placeholders(n, m):
    X = tf.placeholder(tf.float32, shape=(m, None), name = 'X')
    Y = tf.placeholder(tf.float32, shape=(1, None), name = 'Y')
    return X, Y

def initialize_parameters(layers):

	parameters = {}
	prevDims = 62
	for layerNum, numUnits in enumerate(layers):
		parameters['W'+str(layerNum)] = tf.get_variable("W"+str(layerNum), [numUnits,prevDims], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
		parameters['b'+str(layerNum)] = tf.get_variable("b"+str(layerNum), [numUnits, 1], initializer = tf.zeros_initializer())
		prevDims = numUnits

	# output layer
	parameters['W'+str(len(layers))] = tf.get_variable("W"+str(len(layers)), [1, prevDims], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
	parameters['b'+str(len(layers))] = tf.get_variable("b"+str(len(layers)), [1, 1], initializer = tf.zeros_initializer())

	return parameters

def forward_propagation(X, params, nLayers, training=True):
    
    activation = X
    for layerNum in range(nLayers):
    	weight = params['W'+str(layerNum)]
    	bias = params['b'+str(layerNum)]
    	linear = tf.add(tf.matmul(weight, activation), bias)
    	if layerNum != nLayers-1:
    		b = tf.layers.batch_normalization(linear, axis=0, training=training)
    		activation = tf.nn.relu(b)
    		# activation = tf.layers.dropout(activation, rate=0.3, training=training)
    	else:
    		activation = linear

    return activation

def compute_cost(predictions, Y, params):
	logits = tf.transpose(predictions)
	labels = tf.transpose(Y)
	cost = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(labels,logits=logits))

	b = 2
	reg = 0
	for weight in ['W0', 'W1', 'W2', 'W3', 'W4', 'W5']:
		reg += tf.nn.l2_loss(params[weight])
	cost += b*reg

	return cost

def random_mini_batch(X, Y, size):
	indices = np.random.randint(0, X.shape[1], size)
	return X[:, indices], Y[:,indices].reshape((1, size))

def model(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, lr = 1e-4, layers=[150, 100, 50, 25, 12],
          num_epochs = 700, minibatch_size = 512, print_cost = True):
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
    (m, n) = X_train.shape                            # (n: num examples, m : num features)
    costs = []                                        # To drop track of the cost
    trainAccs = []
    devAccs = []
    testAccs = []
    
    X, Y = create_placeholders(n, m)

    # Initialize parameters
    parameters = initialize_parameters(layers)
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    nLayers = len(layers)+1
    predictions = forward_propagation(X, parameters, nLayers)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(predictions, Y, parameters)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    # To track accuracy
    preds = tf.round(tf.sigmoid(forward_propagation(X, parameters, nLayers, training=False)))
    correct_prediction = tf.equal(preds, Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
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
                trainAccs.append(accuracy.eval({X: X_train, Y: Y_train}))
                devAccs.append(accuracy.eval({X: X_dev, Y: Y_dev}))
                testAccs.append(accuracy.eval({X: X_test, Y: Y_test}))


        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Dev Accuracy:", accuracy.eval({X: X_dev, Y: Y_dev}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

    # plot the cost
    plt.plot(trainAccs)
    plt.plot(devAccs)
    plt.plot(testAccs)
    plt.ylabel('Cost')
    plt.xlabel('Iterations (every 5 epochs)')
    plt.title('5-Layer NN Accuracy for Improvement Classification with BN/L2 Reg')
    plt.legend(['Train Accuracy', 'Dev Accuracy', 'Test Accuracy'])
    plt.show()

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

# X_train = getInputData('data/2017clean.csv')
# Y_train = getImprovementLabels('data/2017clean.csv', 'data/2018clean.csv').reshape((1, -1))

# X_test = getInputData('data/2018clean.csv')
# Y_test = getImprovementLabels('data/2018clean.csv', 'data/2019clean.csv').reshape((1, -1))

# n_x = X_test.shape[0]
# dev_indices = np.random.choice(list(range(n_x)), size=int(n_x * 0.5), replace=False)

# X_dev = X_test[dev_indices, :]
# Y_dev = Y_test[:,dev_indices].reshape((1, -1))

# test_indices = list(set(range(n_x)) - set(dev_indices))

# X_test = X_test[test_indices, :]
# Y_test = Y_test[:,test_indices].reshape((1, -1))

X_train = np.loadtxt('data/X_train_classification.txt')
Y_train = np.loadtxt('data/Y_train_classification.txt').reshape((1,-1))
X_dev = np.loadtxt('data/X_dev_classification.txt')
Y_dev = np.loadtxt('data/Y_dev_classification.txt').reshape((1,-1))
X_test = np.loadtxt('data/X_test_classification.txt')
Y_test = np.loadtxt('data/Y_test_classification.txt').reshape((1,-1))

# def stats(a):
#     print("Variance: ", np.var(a))
#     print("Mean: ", np.mean(a))

# stats(Y_train)
# stats(Y_dev)
# stats(Y_test)

parameters = model(X_train.T, Y_train, X_dev.T, Y_dev, X_test.T, Y_test)

