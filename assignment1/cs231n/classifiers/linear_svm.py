import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    # Remember that the loss function is the
    # doees not take into account the score for
    # the correct class so we store it in a variable
    # and simply remove it at the end

    for i in xrange(num_train):
        meet_criteria = 0
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            # remove the correct_class_score
            margin = scores[j] - correct_class_score + 1  # note delta = 1

            # For the the Wj other than the correct one we just add the training example
            if margin > 0:
                meet_criteria += 1
                loss += margin
                dW[:, j] += X[i]
        # For the correct score we add the training example
        # times the number of classes that met the criteria
        # Note the negative as we want to go in the opposite direction
        dW[:, y[i]] -= meet_criteria * X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    dW += 2 * reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W

    """

    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    # compute the loss and the gradient
    num_train = X.shape[0]
    loss = 0.0

    scores_matrix = X.dot(W)  # (N,C)
    correct_scores = scores_matrix[range(len(y)), y]  # (N, ) -> matrix with the correct scores

    correct_classes = [np.arange(y.shape[0]), y]  # shape = (C, N)

    # Create an axis to we can broadcast
    # https://mlxai.github.io/2017/01/06/vectorized-implementation-of-svm-loss-and-gradient-update.html

    # Element i, j of scores_matrix contain the Loss (Li) for sample i and class j
    # So to each of them I need to remove the loss of the correct class which is contained on correct scores
    margin_matrix = scores_matrix - correct_scores[:, np.newaxis] + 1  # (N,C)

    # We do this because at this point  the correct scores are equal to the delta (which = 1, see above)
    margin_matrix[np.arange(num_train), y] = 0

    margin_matrix = np.maximum(margin_matrix, 0)  # clamp to zero as per formula

    loss = np.sum(margin_matrix)  # The overall loss is the sum of the individual loss over classes and samples

    # Binarize the margin matrix to be able to count those that are not 0
    # This gives a binary matrix

    classes_beyond_margin = (margin_matrix > 0).astype(np.float64)  # shape (N, C)

    # Some over rows, so we have the number of Li  that are > 0 over  the classes
    # classes_beyond_margin:

    #                C
    #       +--+---------+
    #       |  |         |
    # N     |  |         |
    #       |  |         |
    #       |  |         |
    #       +--+---------+
    #          l_i: Loss count for class j

    # Coontains for each class the number of missclassfied samples for the whole set.
    # Equivalent of meet_criteria on the non-vectorized version but for the whole datase
    num_classes_beyond_margin = np.sum((classes_beyond_margin), axis=1)  # shape N

    # This replaces the 0 for the number of incorrect classes (larger than zero)

    classes_beyond_margin[correct_classes] = -num_classes_beyond_margin

    # X = (N, D)
    # classes_beyond_margin = N, C
    # dW = D, C
    dW = X.T.dot(classes_beyond_margin)

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################

    loss /= num_train
    dW /= num_train

    # Regularization
    loss += reg * np.sum(W * W)
    dW += reg * W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
