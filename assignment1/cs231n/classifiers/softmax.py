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
  num_train=X.shape[0]
  num_class=W.shape[1]
  scores=X.dot(W)
  scores_max=np.argmax(scores,axis=1)
  #print scores_max.shape
  scores_max=np.reshape(scores_max,(num_train,1))
  scores_exp=np.exp(scores-scores_max)
  sum=np.sum(scores_exp,axis=1)
 
  for i in xrange(num_train):
      loss -= np.log(scores_exp[i,y[i]]/sum[i])
  
  loss /= num_train
  loss += 2*reg*np.sum(W*W)
  
  
  for i in xrange(num_train):
      for j in xrange(num_class):
          ok=0       
          if j==y[i]:
              ok=1
          dW[:,j] += (scores_exp[i,j]/sum[i]-ok)*X[i,:]
  dW /= num_train
  dW += reg*W
  
#  prob = scores_exp / sum.reshape(sum.shape[0], 1)
  
#  for j in xrange(W.shape[1]): # j in C
#      acc = 0
#      for i in xrange(X.shape[0]):  # i in N  prob N * C
#          true = 0
#          if y[i] == j:
#              true = 1
##          if flag == 0:
##              acc = X[i,:] * (prob[i, j] - true)
##              flag = 1
##          else:
#          acc += X[i,:] * (prob[i, j] - true)
#      dW[:, j] = acc / X.shape[0]
#      
#  dW += reg*W         
      
          
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train=X.shape[0]
  #num_class=W.shape[0]
  scores=X.dot(W)
  scores_max=np.argmax(scores,axis=1)
  #print scores_max.shape
  scores_max=np.reshape(scores_max,(num_train,1))
  scores_exp=np.exp(scores-scores_max)
  sum=np.sum(scores_exp,axis=1)
 
  ans=scores_exp/np.reshape(sum,(num_train,1) )
  mask=np.zeros_like(ans)
  
  mask[range(num_train),y]=1
  
  loss = -(np.sum(mask*np.log(ans)))
  loss /= num_train
  loss += 0.5*reg*np.sum(W*W)
  
  ans[range(num_train),y] -= 1
  dW=X.T.dot(ans)
  dW /= num_train
  dW += reg*W
  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

