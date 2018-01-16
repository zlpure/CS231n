import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  x_temp=x.reshape((x.shape[0],-1))
  out=np.dot(x_temp,w)+b
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  db=np.sum(dout,axis=0)
  
  x_temp=x.reshape((x.shape[0],-1))
  dw=np.dot(x_temp.T,dout)
  
  dx=dout.dot(w.T)
  dx=np.reshape(dx,x.shape)  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.maximum(0,x)
  #out=(x>=0)*x
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  temp = (x>=0)
  dx = dout*temp
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    sample_mean = np.mean(x,axis=0)
    sample_var = np.var(x,axis=0)

    x_normal = (x-sample_mean[np.newaxis,:])/np.sqrt(sample_var[np.newaxis,:]+eps)    
    out = gamma[np.newaxis,:]*x_normal+beta[np.newaxis,:]
    
    cache=(x, gamma, beta, x_normal,sample_mean,sample_var,eps)
    
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################   
    sample_mean = np.mean(x,axis=0)
    sample_var = np.var(x,axis=0)
    
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    
    x_normal = (x-running_mean[np.newaxis,:])/np.sqrt(running_var[np.newaxis,:]+eps)    
    out = gamma[np.newaxis,:]*x_normal+beta[np.newaxis,:]
    cache=(x, gamma, beta, x_normal,sample_mean,sample_var,eps)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  x, gamma, beta, x_normal,sample_mean,sample_var,eps = cache
  N=x.shape[0] #is hang,bu shi lie
  
  dx_ba = dout * gamma
  
  dgamma = np.sum(dout * x_normal,axis=0)
  
  dbeta = np.sum(dout,axis=0)
  
  dderta = np.sum(dx_ba*(x-sample_mean)*(-0.5*(sample_var+eps)**(-1.5)),axis=0)
  
  dmu = np.sum(dx_ba*(-1*(sample_var+eps)**(-0.5)),axis=0) + dderta*(np.sum((-2)*(x-sample_mean),axis=0)/N)
  
  dx = dx_ba*( (sample_var + eps)**(-0.5) )+ dderta*(2*(x-sample_mean)/N) + dmu/N
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  x, gamma, beta, x_normal,sample_mean,sample_var,eps = cache 
  
  dgamma = np.sum(dout * x_normal,axis=0)
  dbeta = np.sum(dout,axis=0)
  
  dx_ba = dout * gamma
  dtemp1 = 1/np.sqrt(sample_var+eps)
  dtemp2 = (x - sample_mean)*dtemp1
  
  dx=(dx_ba-np.mean(dx_ba,axis=0)-np.mean(dx_ba*dtemp2,axis=0)*dtemp2)*dtemp1  
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    mask = (np.random.rand(*x.shape)<p)/p
    out = x*mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    #out = x*(np.random.rand(*x.shape)<p)/p
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx = dout*mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  N,C,H,W=x.shape
  F,_,HH,WW=w.shape
  s,p=conv_param['stride'],conv_param['pad']
  
  x_pad=np.pad(x,((0,0),(0,0),(p,p),(p,p)),'constant') #default is 0
  
  out_h = (H - HH + 2*p) / s + 1
  out_w = (W - WW + 2*p) / s + 1
  out=np.zeros((N,F,out_h,out_w))
  
  #x_col=np.zeros((C*HH*WW,out_h*out_w))
  #w_row=np.zeros((F,C*HH*WW))
  
  for img_num in xrange(N):
      for fil_num in xrange(F):
          conv_img=np.zeros((out_h,out_w))
          for col_num in xrange(C):
              img=x_pad[img_num,col_num,:,:]
              fil=w[fil_num,col_num,:,:]
              for conv_row in xrange(out_h):
                  row_strat=conv_row*s
                  row_end=row_strat+HH
                  for conv_col in xrange(out_w):
                      col_strat=conv_col*s
                      col_end=col_strat+WW
                      conv_img[conv_row,conv_col]+=np.sum(img[row_strat:row_end,col_strat:col_end]*fil)
          out[img_num,fil_num,:,:]=conv_img+b[fil_num]
      

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x,w,b,conv_param=cache
  N,C,H,W=x.shape
  F,_,HH,WW=w.shape
  _,_,out_h,out_w=dout.shape
  s,p=conv_param['stride'],conv_param['pad']
  
  x_pad=np.pad(x,((0,0),(0,0),(p,p),(p,p)),'constant') #default is 0
 
  #dw:F*C*HH*WW
  dw=np.zeros(w.shape)
  for fil_num in xrange(F):
      for col_num in xrange(C):
          w_temp=np.zeros((HH,WW))
          for img_num in xrange(N):
              img=x_pad[img_num,col_num,:,:]
              delta=dout[img_num,fil_num,:,:]
              for w_row in xrange(HH):
                  for w_col in xrange(WW):
                     temp=img[w_row:w_row+out_h*s:s,w_col:w_col+out_w*s:s]
                     w_temp[w_row,w_col]+=np.sum(temp*delta)
          dw[fil_num,col_num,:,:]+=w_temp
 

  #dx:N*C*H*W
  dx=np.zeros(x.shape)
  for img_num in xrange(N):
      for col_num in xrange(C):
          for fil_num in xrange(F):
              x_temp=np.dot(dout[img_num,fil_num,:,:].reshape(-1,1),w[fil_num,col_num,:,:].reshape(1,-1))
              x_temp.shape=(out_h,out_w,HH,WW)
              #print x_temp.shape
              dx_temp=np.zeros((H+2*p,W+2*p))
              #print dx_temp.shape
              for conv_row in xrange(out_h):
                  row_strat=conv_row*s
                  row_end=row_strat+HH
                  for conv_col in xrange(out_w):
                      col_strat=conv_col*s
                      col_end=col_strat+WW
                      dx_temp[row_strat:row_end,col_strat:col_end]+=x_temp[conv_row,conv_col,:,:]
              dx[img_num,col_num,:,:]+=dx_temp[p:-p,p:-p]
       
  #db:f
  db=np.sum(dout,axis=(0,2,3))
              
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db

def max_pool_forward_naive1(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
  
  N, C, H, W = x.shape
  
  H_new = 1 + (H - pool_height) / stride   
  W_new = 1 + (W - pool_height) / stride
  
  out = np.zeros((N, C, H_new, W_new))
  
  for n in xrange(N):
      for s1 in xrange(H_new):
          for s2 in xrange(W_new):
              out[n, :, s1, s2] = np.max(x[n, :, s1 * stride : (s1 * stride + pool_height), s2 * stride : (s2 * stride) + pool_width], (1,2))
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  #out = np.zeros((N, C, H_new, W_new))
  
  return out, cache

def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  N,C,H,W=x.shape
  pool_height=pool_param['pool_height']
  pool_width=pool_param['pool_width']
  stride=pool_param['stride']
  out_height=(H-pool_height)/stride+1
  out_width=(W-pool_width)/stride+1
  
  out=np.zeros((N,C,out_height,out_width))
  for img_num in xrange(N):
      for col_num in xrange(C):
          for pool_row in xrange(out_height):
              row_start=pool_row*stride
              row_end=row_start+pool_height
              for pool_col in xrange(out_width):
                  col_start=pool_col*stride
                  col_end=col_start+pool_width
                  kernrl=x[img_num,col_num,row_start:row_end,col_start:col_end]
                  out[img_num,col_num,pool_row,pool_col]=kernrl.max()
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x,pool_param=cache
  N,C,H,W=x.shape
  pool_height=pool_param['pool_height']
  pool_width=pool_param['pool_width']
  stride=pool_param['stride']
  _,_,out_h,out_w=dout.shape  
  
  dx=np.zeros(x.shape)
  for img_num in xrange(N):
      for col_num in xrange(C):
          for pool_row in xrange(out_h):
              row_start=pool_row*stride
              row_end=row_start+pool_height
              for pool_col in xrange(out_w):
                  col_start=pool_col*stride
                  col_end=col_start+pool_width
                  kernel=x[img_num,col_num,row_start:row_end,col_start:col_end]
                  index=np.unravel_index(kernel.argmax(),kernel.shape)
                  dx[img_num,col_num,row_start+index[0],col_start+index[1]]+=dout[img_num,col_num,pool_row,pool_col]
                  
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  N,C,H,W=x.shape
  x = x.transpose(0,2,3,1).reshape(N*H*W,C)
  
  out,cache=batchnorm_forward(x,gamma,beta,bn_param)
  out = out.reshape(N,H,W,C).transpose(0,3,1,2)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  N,C,H,W=dout.shape
  dout = dout.transpose(0,2,3,1).reshape(N*H*W,C)
  
  dx,dgamma,dbeta=batchnorm_backward_alt(dout,cache)
  dx = dx.reshape(N,H,W,C).transpose(0,3,1,2)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx


def softmax_loss_temp(x, y,alpha=1.0,belta=0.1):
  c = x.shape[1]
  x_temp = alpha*x[:,0::c]
  print x_temp.shape 
  for i in range(c):
      x_temp[:,i] = x_temp[:,i]+0.5*belta*np.sum(x[:,c*i+1:c*(i+1)],axis=1)
  print x_temp.shape  
  probs = np.exp(x_temp - np.max(x_temp, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  #print sum_fenzi.shape
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx  
  
def fun_temp(x,y):
  c = x.shape[1]
  temp = np.repeat(x,c,axis=1)
  N = x.shape[0]
  sum_fenzi = x.copy()
  for i in range(c):
      sum_fenzi[:,i] = np.sum(temp[:,c*i:c*(i+1)])
  print sum_fenzi.shape
  loss = np.sum(sum_fenzi[np.arange(N), y]) / N
  
  dx = sum_fenzi.copy()
  dx[np.arange(N), y] *= c
  dx /= N
  return loss, dx
