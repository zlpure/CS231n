import numpy as np
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class ConvNet2(object):
  """
  A convolutional network with the following architecture:
  
  [conv-spatial batch normalization - relu - max_pool] * N - conv - relu - [affine-vanilla batch normalization] * M -[softmax or SVM]
  
  #max_pool size is 2
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  def __init__(self,  hidden_dims, num_filters, input_dim=(3, 32, 32), filter_size=3, 
               num_classes=10, weight_scale=1e-1, reg=0.0, dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - num_filters: A list of integers giving the size of each conv layer.
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.num_filters = num_filters
    self.filter_size = filter_size
    
    # TODO: Initialize weights and biases for the conplex convolutional    #
    # network. Weights should be initialized from a Gaussian with standard  #
    # deviation equal to weight_scale; biases should be initialized to zero. #
    # All weights and biases should be stored in the dictionary self.params. #
    self.conv_num=len(num_filters)-1
    num_filters.insert(0,3)
    
    self.affi_num=len(hidden_dims)+1
    
    for i in xrange(self.conv_num):
        index = i+1
        self.params['W'+str(index)]=weight_scale*np.random.randn(num_filters[index],num_filters[index-1],filter_size,filter_size)
        self.params['b'+str(index)]=np.zeros(num_filters[index])
        self.params['gamma'+str(index)]=np.ones(num_filters[index])
        self.params['beta'+str(index)]=np.zeros(num_filters[index])
        
    self.params['W'+str(self.conv_num+1)]=weight_scale*np.random.randn(num_filters[self.conv_num+1],num_filters[self.conv_num],filter_size,filter_size)
    self.params['b'+str(self.conv_num+1)]=np.zeros(num_filters[self.conv_num+1])
    
    temp=hidden_dims
    a=(32/(2**self.conv_num))**2*num_filters[-2]
    temp.insert(0,a)
    temp.append(num_classes)
    
    
    for i in xrange(self.affi_num):
        index=i+1
        self.params['W'+str(self.conv_num+1+index)]=weight_scale*np.random.randn(temp[index-1],temp[index])
        self.params['b'+str(self.conv_num+1+index)]=np.zeros(temp[index])
        self.params['gamma'+str(self.conv_num+1+index)]=np.ones(temp[index])
        self.params['beta'+str(self.conv_num+1+index)]=np.zeros(temp[index])
    
    self.bn_params = []
    self.bn_params = [{'mode': 'train'} for i in xrange(self.conv_num+self.affi_num)]
    
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype) 
      
    
  def loss(self, X, y=None):
      # pass conv_param to the forward pass for the convolutional layer
      filter_size = self.filter_size
      conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

      # pass pool_param to the forward pass for the max-pooling layer
      pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        
      X = X.astype(self.dtype)
      mode = 'test' if y is None else 'train'     
        
      for bn_param in self.bn_params:
          bn_param[mode] = mode
    
      scores = None 
        
      cache={}
      score_temp=X
      for i in xrange(self.conv_num):
          out,cache['layer'+str(i+1)]=conv_batchnorm_relu_pool_forward(score_temp,self.params['W'+str(i+1)],self.params['b'+str(i+1)],self.params['gamma'+str(i+1)],self.params['beta'+str(i+1)],conv_param,pool_param,self.bn_params[i])
          score_temp=out
      
      s,cache['layer'+str(self.conv_num+1)]=conv_relu_forward(out,self.params['W'+str(self.conv_num+1)],self.params['b'+str(self.conv_num+1)],conv_param)
      #print s.shape
      #print self.params['W'+str(self.conv_num+2)].shape
      #print self.bn_params[self.conv_num+self.affi_num]
      #print cache['layer'+str(self.conv_num)]
      for i in xrange(self.affi_num):
          index=self.conv_num+1+i+1
          scores,cache['layer'+str(index)]=affine_batchnorm_relu_forward(s,self.params['W'+str(index)],self.params['b'+str(index)],self.params['gamma'+str(index)],self.params['beta'+str(index)],self.bn_params[index-2])
          s=scores
      
      if mode == 'test':
          return scores

      loss, grads = 0.0, {}
      loss,dx=softmax_loss_temp(scores,y)
      
      for i in xrange(self.conv_num+1):
        loss += 0.5*self.reg*np.sum(np.square(self.params['W'+str(i+1)]))
      for i in xrange(self.affi_num):
        index=self.conv_num+1+i+1  
        loss += 0.5*self.reg*np.sum(np.square(self.params['W'+str(index)]))
       
      for i in xrange(self.affi_num-1,-1,-1):
          index=self.conv_num+1+i+1
          dscore,grads['W'+str(index)],grads['b'+str(index)],grads['gamma'+str(index)],grads['beta'+str(index)]=affine_batchnorm_relu_backward(dx,cache['layer'+str(index)])
          dx=dscore
          grads['W'+str(index)] += self.reg*grads['W'+str(index)]
          
      dout,grads['W'+str(self.conv_num+1)],grads['b'+str(self.conv_num+1)]=conv_relu_backward(dscore, cache['layer'+str(self.conv_num+1)])
      grads['W'+str(self.conv_num+1)] +=self.reg*grads['W'+str(self.conv_num+1)]      
      
      for i in xrange(self.conv_num-1,-1,-1):
          dd,grads['W'+str(i+1)],grads['b'+str(i+1)],grads['gamma'+str(i+1)],grads['beta'+str(i+1)]=conv_batchnorm_relu_pool_backward(dout,cache['layer'+str(i+1)])
          dout=dd
          grads['W'+str(i+1)] += self.reg*grads['W'+str(i+1)]
        
      return loss,grads
        
        
        
        
        
        
        
        
'''
        
        
     
def affine_batchnorm_relu_forward(x,w,b,gamma, beta, bn_param):
    a, fc_cache = affine_forward(x, w, b)
    a2, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(a2)
    cache = (fc_cache,bn_cache,relu_cache)
    return out,cache

def affine_batchnorm_relu_backward(dout,cache):
    fc_cache,bn_cache,relu_cache = cache
    da = relu_backward(dout,relu_cache)
    da2,dgamma,dbeta = batchnorm_backward(da,bn_cache)
    dx, dw, db = affine_backward(da2, fc_cache)
    return dx,dw,db,dgamma,dbeta

def conv_batchnorm_relu_pool_forward(x, w, b,gamma, beta, conv_param, pool_param,bn_param):       
    a,conv_cache = conv_forward_fast(x,w,b,conv_param)
    a2,bn_cache=spatial_batchnorm_forward(a,gamma,beta,bn_param)
    a3,relu_cache=relu_forward(a2)
    out,pool_cache=max_pool_forward_fast(a3,pool_param)
    cache=(conv_cache,bn_cache,relu_cache,pool_cache)
    return out,cache

def conv_batchnorm_relu_pool_backward(dout,cache):
    conv_cache,bn_cache,relu_cache,pool_cache=cache
    da=max_pool_backward_fast(dout, pool_cache)
    da2=relu_backward(da,relu_cache)
    da3,dgamma,dbeta=spatial_batchnorm_backward(da2, bn_cache)
    dx, dw, db = conv_backward_fast(da3, conv_cache)
    return dx, dw, db,dgamma,dbeta
'''