from __future__ import print_function
from typing import Dict

import numpy as np
from past.builtins import xrange


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params: Dict[str, np.array] = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    @classmethod
    def _softmax_loss(cls, scores, y):
        """
        Softmax loss function, vectorized version.
        Inputs have dimension D, there are C classes, and we operate on minibatches
        of N examples.

        Inputs:
        - scores: A numpy array of shape (N, C) containing scores per (training_sample, classes)
        - y: A numpy array of shape (N,) containing training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.

        Returns a tuple of:
        - loss as single float
        - gradient with respect to the scores variables

        """
        # Initialize the loss and gradient to zero.
        loss = 0.0
        num_train = scores.shape[0]

        # scores = np.dot(X, W) + b -> Vectorial
        scores_matrix = scores  # (N, D) x (D,C) = NxC -> scores samples per class
        correct_scores_matrix = scores_matrix[range(len(y)), y]  # (N, ) -> vector with the correct scores
        loss_vect = np.log(np.exp(scores_matrix).sum(axis=1)) - correct_scores_matrix  # (N, ) loss per sample
        loss = loss_vect.sum()

        # Normalization
        scores_matrix_norm = scores_matrix - np.max(scores_matrix, axis=1)[:, np.newaxis]  # NxC

        # NxC (Probs per class)
        p_matrix = np.exp(scores_matrix_norm) / np.sum(np.exp(scores_matrix_norm), axis=1)[:, np.newaxis]

        dscores = p_matrix
        dscores[range(num_train), y] -= 1
        dscores /= num_train

        loss /= num_train

        return loss, dscores

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################

        def f(x):  # relu
            return np.maximum(0, x)

        # Fully connected + ReLu
        fc1 = X.dot(W1) + b1
        h1 = f(fc1)  # (N, D) x (D,W1.shape[1]) = NxW1.shape[0] -> scores samples per activation

        # scores = a2 - np.max(a2, axis=1)[:, np.newaxis]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # If the targets are not given then jump out, we're done
        scores = h1.dot(W2) + b2
        if y is None:
            return scores

        # Compute the loss

        # loss = 0.0
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        # Fully connected + softmax
        loss, dscores = TwoLayerNet._softmax_loss(scores, y)
        loss += reg * np.sum(W2 * W2)
        loss += reg * np.sum(W1 * W1)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # Backward pass: compute gradients
        grads = {}

        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################

        # Softmax is a function of two variables
        # the hidden layer and W2
        # We need to backpropagate into both. For a simple SM classfier this wouldn't be  necessary
        # as the  one of the input is the actual training data but in this case is another
        # Layer and thus needs to be backpropagated in order to get the derivatives of the previous layer

        dW2 = np.dot(h1.T, dscores)
        db2 = np.sum(dscores, axis=0)

        # next backprop into hidden layer
        dhidden = np.dot(dscores, W2.T)
        dhidden[h1 <= 0] = 0

        # finally into W,b
        dW1 = np.dot(X.T, dhidden)
        db1 = np.sum(dhidden, axis=0)

        # Regularization
        dW2 += 2 * reg * W2
        dW1 += 2 * reg * W1

        # Store them into the dictionary
        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W1'] = dW1
        grads['b1'] = db1

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving train Ning data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in xrange(num_iters):

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            b_index = np.random.choice(np.arange(y.shape[0]), replace=True, size=batch_size)

            X_batch = X[b_index, :]
            y_batch = y[b_index]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            for p in grads:
                self.params[p] += -learning_rate * grads[p]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        y_pred = np.argmax(self.loss(X), axis=1)
        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################

        return y_pred
