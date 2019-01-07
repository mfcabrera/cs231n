import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in xrange(num_train):
        scores = X[i].dot(W)   # 1,D x (D,C) = 1xC (score per class)
        correct_class_score = scores[y[i]]
        scores_norm = scores - np.max(scores)
        p = np.exp(scores_norm) / np.sum(np.exp(scores_norm))

        loss += -correct_class_score + np.log(np.exp(scores).sum())
        for c in xrange(num_classes):
            dW[:, c] += p[c] * X[i]  # (1, D)

        dW[:, y[i]] += -X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.

    loss /= num_train
    loss += reg * np.sum(W * W)
    # Same for dW
    dW /= num_train
    dW += 2 * reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]
    mask = np.zeros((num_train, num_classes))  # NxC
    mask[np.arange(y.shape[0]), y] = 1

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    scores_matrix = X.dot(W)  # (N, D) x (D,C) = NxC -> scores samples per class
    correct_scores_matrix = scores_matrix[range(len(y)), y]  # (N, ) -> vector with the correct scores
    loss_vect = np.log(np.exp(scores_matrix).sum(axis=1)) - correct_scores_matrix  # (N, ) loss per sample
    loss = loss_vect.sum()

    scores_matrix_norm = scores_matrix - np.max(scores_matrix, axis=1)[:, np.newaxis]
    p_matrix = np.exp(scores_matrix_norm) / np.sum(np.exp(scores_matrix_norm), axis=1)[:, np.newaxis]

    grad_tensor = X[:, :, np.newaxis] * p_matrix[:, np.newaxis, :]  # (NxDx1) * (Nx1xC) =  (N x D) x C

    dW += grad_tensor.sum(axis=0)  # DxC -> sum all out all samples
    dW += -X.T.dot(mask)  # (DxN) x (NxC) = DxC

    loss /= num_train
    loss += reg * np.sum(W * W)
    # Same for dW
    dW /= num_train
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
