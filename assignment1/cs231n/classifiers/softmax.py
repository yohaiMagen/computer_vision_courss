import numpy as np
from random import shuffle

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
  score = np.exp(X.dot(W))
  normalization = np.sum(score, axis=1)
  for i in range(y.shape[0]):
    loss += -np.log(score[i, y[i]] / normalization[i])
    # dW[:, y[i]] -= ((X[i, :] * (score[i, y[i]] * normalization[i] - score[i, y[i]]**2)) / normalization[i]**2) / np.log(10)*loss
    dW[:, y[i]] += X[i, :] * (-1 + score[i, y[i]] / normalization[i])
    for j in np.delete(np.arange(10), y[i]):
        # dW[:, j] += ((score[i, y[i]] * score[i, j] *X[i, :]) / normalization[i]**2) / np.log(10)*loss
        # dW[:, j] += X[i, :]
        dW[:, j] += X[i, :] * score[i, j] / normalization[i]
  loss /= y.shape[0]
  loss += reg * np.sum(W*W)

  dW /= y.shape[0]
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train =X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  score = np.exp(X.dot(W))
  normalization = np.sum(score, axis=1)

  loss = np.sum(-np.log(score[np.arange(num_train), y] / normalization)) / num_train + reg * (np.sum(W*W))

  A = score / normalization[:, np.newaxis]
  B = np.zeros((num_train, W.shape[1]))
  B[np.arange(num_train), y] = -1
  dW = X.T.dot(A + B) / num_train + reg * W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

