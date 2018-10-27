# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=protected-access
"""Recurrent layers and their base classes.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.checkpointable import base as checkpointable
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

"""
                      /-> | 0 1 | -\
|   |         | 0 |  /    | 1 0 |   \  
| I |  ====>  | 1 | /-->  | 0 1 |    \-> | 1 | ==>
| n |         | 0 |/      | 1 0 |     \  | 0 | ==>
| p |  ====>  | 1 |---->  | 0 1 |      ->| 1 | ==>
| u |         | 0 |\      | 1 0 |     /  | 0 | ==>
| t |  ====>  | 1 | \-->  | 0 1 |    /-> | 1 | ==>
|   |         | 0 |  \    | 0 1 |   /
                      \-> | 1 0 | -/


             kern_1         kern_2        kern_3        

"""

class ESN(ESN_kernel):
    """"
ESN reservoir functional wrapper
    # Arguments
        units: Positive integer, number of units in the reservoir space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        init_kern_1: Initializer for the input adapter's `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        init_kern_2: Initializer for the real reservoir's `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        init_kern_3: Initializer for the output adapter's `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        init_recur_kern_2: Initializer for the recurrent units inside the reservoir's
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        init_out_fb: Initializer for the output's feedback to kern_1 weights matrix.
            (see [initializers](../initializers.md)).
        init_bias: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        connectivity: Float between 0 and 1.
            Connectivity percentage between inner reservoir units   .
        reg_kern_1: Regularizer function applied to
            the input adapter's `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        reg_kern_3: Regularizer function applied to
            the output adapter's `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        reg_out_fb: Regularizer function applied to
            the output's feedback weights matrix.
            (see [regularizer](../regularizers.md)).
        reg_bias: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        constraint_kern_1: Constraint function applied to
            the input adapter weights matrix
            (see [constraints](../constraints.md)).
        constraint_kern_3: Constraint function applied to
            the output adapter weights matrix
            (see [constraints](../constraints.md)).
        constraint_recur_kern_2: Constraint function applied to
            the reservoir's recurrent weights matrix
            (see [constraints](../constraints.md)).
        constraint_out_fb: constraint function applied to
            the output's feedback weights matrix.
        constraint_bias: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        in_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recur_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        train_kern_1: Boolean( default False).
            if True, input adapter's weight matrix is trained.
        train_kern_3: Boolean( default False).
            if True, output adapter's weight matrix is trained
        train_out_fb: Boolean( default False).
            if True, output feedback's weight matrix is trained   
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
    """

    @interfaces.legacy_recurrent_support
    def __init__(self,
             units,
               activation='tanh',
               init_kern_1='RandomUniform',
               init_kern_2='RandomUniform',
               init_kern_3='RandomUniform',
               init_recur_kern_2='orthogonal',
               init_out_fb='orthogonal',
               init_bias='zeros',
               connectivity=None,
               reg_kern_1=None,
               reg_kern_3=None,
               reg_out_fb=None,
               reg_bias=None,
               constraint_kern_1=None,
               constraint_kern_3=None,
               constraint_recur_kern_2=None,
               constraint_out_fb=None,
               constraint_bias=None,
               in_dropout=None,
               recur_dropout=None,
               train_kern_1=False,
               train_kern_3=False,
               train_out_fb=False,
               use_out_fb=False,
               use_dropout_mask=False,
               use_recur=False,
               use_clock=False,
               clock_rate=None,
               data_format=None,
               **kwargs):
        if K.backend() == 'theano' and (dropout or  recur_dropout):
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and ` recur_dropout` to 0, '
                'or use the TensorFlow backend.')
            
        cell = ESN_kernel(units,
               units,
               activation='tanh',
               init_kern_1='RandomUniform',
               init_kern_2='RandomUniform',
               init_kern_3='RandomUniform',
               init_recur_kern_2='orthogonal',
               init_out_fb='orthogonal',
               init_bias='zeros',
               connectivity=None,
               reg_kern_1=None,
               reg_kern_3=None,
               reg_out_fb=None,
               reg_bias=None,
               constraint_kern_1=None,
               constraint_kern_3=None,
               constraint_recur_kern_2=None,
               constraint_out_fb=None,
               constraint_bias=None,
               in_dropout=None,
               recur_dropout=None,
               train_kern_1=False,
               train_kern_3=False,
               train_out_fb=False,
               use_out_fb=False,
               use_dropout_mask=False,
               use_recur=False,
               use_clock=False,
               clock_rate=None,
               data_format=None,
               **kwargs) 
        super(ESN, self).__init__(cell,
                                        use_bias=use_bias,
                                        return_sequences=return_sequences,
                                        return_state=return_state,
                                        go_backwards=go_backwards,
                                        stateful=stateful,
                                        unroll=unroll,
                                        **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, prev_states):
        output_fb = self.prev_states[0]
        recur_output = self.prev_states[1]
        if self.cell.use_clock is False:
            input = K.dot(self.inputs * self.cell.in_dropout_mask,
                          self.cell.kern_1)  
        else:
            input = K.dot(self.inputs * self.cell.in_dropout_mask * self.cell.clock_kernel,
                          self.cell.kern_1)

        if self.cell.use_out_fb is not False:
             x = K.dot(_pad(self.cell.out_fb_kern * output_fb, (self.cell.in_row, self.cell.in_col)),
                       self.inputs)
             input = K.bias_add(x, input)
                    
        if self.cell.use_recur is True:
            reservoir_output_1 = recur_output * self.cell.recur_dropout_mask
            reservoir_output_2 = K.dot(input, self.cell.kern_2)
            reservoir_output = K.bias_add(kern_output_1
                                     kern_output_2)
        else:
            reservoir_output = K.dot(input, self.cell.kern_2)
               
        output =  K.dot(reservoir_output, kern_3)
            
        return output, [output, reservoir_output]                                                                                    

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer':
                      initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer':
                      initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer':
                      regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer':
                      regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer':
                      regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint':
                      constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(SimpleRNN, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config:
            config.pop('implementation')
        return cls(**config)




@tf_export('keras.layers.ESN_kernel')
class ESN_kernel(layer):
  """ ESN kernel constructor
 
     the philosophy behind this method of construction is making adaptors (kern_1,kern_3)
     so padding of the input is not necessary and to scale down the output to be
     compatible with next layers and this has proved more suitable for my application.

  Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
          Default: hyperbolic tangent (`tanh`).
          If you pass `None`, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
          used for the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel`
          weights matrix,
          used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
          the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
          the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      kernel_constraint: Constraint function applied to
          the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
          the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      dropout: Float between 0 and 1.
          Fraction of the units to drop for
          the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
          Fraction of the units to drop for
          the linear transformation of the recurrent state.
  """
#added function for kernel normalization
#delete all regularizers
#set training = False
  def __init__(self,
               units,
               activation='tanh',
               init_kern_1='RandomUniform',
               init_kern_2='RandomUniform',
               init_kern_3='RandomUniform',
               init_recur_kern_2='orthogonal',
               init_out_fb='orthogonal',
               init_bias='zeros',
               connectivity=None,
               reg_kern_1=None,
               reg_kern_3=None,
               reg_out_fb=None,
               reg_bias=None,
               constraint_kern_1=None,
               constraint_kern_3=None,
               constraint_recur_kern_2=None,
               constraint_out_fb=None,
               constraint_bias=None,
               in_dropout=None,
               recur_dropout=None,
               train_kern_1=False,
               train_kern_3=False,
               train_out_fb=False,
               use_out_fb=False,
               use_dropout_mask=False,
               use_recur=False,
               use_clock=False,
               clock_rate=None,
               data_format=None,
               **kwargs):
    super(SimpleRNNCell, self).__init__(**kwargs)
    self.units = units

    self.activation = activations.get(activation)

    self.init_kern_1 = initializers.get(init_kern_1)
    self.init_kern_2 = initializers.get(init_kern_2)
    self.init_kern_3 = initializers.get(init_kern_3)
    self.init_recur_kern_2 = initializers.get(init_recur_kern_2)
    self.init_out_fb = initializers.get(init_out_fb)
    self.init_bias = initializers.get(init_bias)

    if len(connectivity) is not 3:
       self.connectivity_1 = 1.
       self.connectivity_2 = 1.
       self.connectivity_3 = 1.
    else:     
       self.connectivity_1 =  min(1., max(0.,connectivity[0]))
       self.connectivity_2 =  min(1., max(0.,connectivity[1]))
       self.connectivity_3 =  min(1., max(0.,connectivity[2]))
    
    self.reg_kern_1 = regularizers.get(reg_kern_1)
    self.reg_kern_3 = regularizers.get(reg_kern_3)
    self.reg_out_fb = regularizers.get(reg_out_fb)
    self.reg_bias = regularizers.get(reg_bias)
    
    self.constraint_kern_1 = constraints.get(constraint_kern_1)
    self.constraint_kern_3 = constraints.get(constraint_kern_3) 
    self.constraint_recur_kern_2 = constraints.get(constraint_recur_kern_2)
    self.constraint_out_fb = constraints.get(constraint_out_fb)
    self.constraint_bias = constraints.get(constraint_bias)
                                       
    self.in_dropout = min(1., max(0.,in_dropout))
    self.recur_dropout = min(1., max(0.,recur_dropout))
    
    self.train_kern_1 = train_kern_1
    self.train_kern_3 = train_kern_3

    self.clock = clock  
    self.clock_rate = clock_rate

    self.in_dropout_mask = None
    self.recur_dropout_mask = None
    self.state_size = self.units
    self.output_size = self.units
    self.tf_data_format = self.data_format
    self.clock_kernel = None
   
  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    in_row, in_col, out_col, input_dim = _preprocess_format(input_shape,tf_data_format)
     
    if( (in_row * in_col) +
        (in_row * out_col) => units ):      # no of units must be enough to at least create the first and last layer 
         raise ValueError('No. of units must be at least '
                          'more than twice the input matrix size')
     
    self.kern_shape_1 = (in_row, in_col, input_dim)
    self.kern_1 = self.add_weight(                                           #layer opposite to input
        shape=kern_shape_1,
        name='kernel1',
        initializer=self.init_kern_1,
        regularizer=reg_kern_1,
        constraint=self.constraint_kern_1,
        trainable=train_kern_1)

    l_2_col = get_layer_dim(in_row, in_col, out_col, units)
    self.kern_shape_2 = (in_col, l_2_col, input_dim)
    self.kern_2 = self.add_weight(                                           #real reservoir 
        shape=kern_shape_2,
        name='kernel2',
        initializer=self.init_kern_2,
        trainable=False)
    if self.use_recur:
        self.recur_kern_2 = self.add_weight(                               
             shape=kern_shape_2,
             name='recurrent_kernel_2',
             initializer=self.init_recur_kern_2,
             constraint=self.constraint_recur_kern_2,
             trainable=False)
        
    self.kern_shape_3 = (l_2_col, out_col, input_dim)                              
    self.kern_3 = self.add_weight(                                           #layer opposite to output
        shape=kern_shape_3,
        name='kernel3',
        initializer=self.init_kern_3,
        regularizer=reg_kern_3,
        constraint=self.constraint_kern_3,
        trainable=train_kern_3)
    
    self.kern_shape_out_fb = (in_row, out_col, input_dim)
    if self.use_out_fb = True:
        self.out_fb_kern =  self.add_weight(                               
           shape=kern_shape_out_fb,
           name='output_feedback',
           initializer=self.init_out_fb,
           regularizer=self.reg_out_fb,
           constraint=self.constraint_out_fb,
           trainable=Train_out_fb)

    else:
      self.bias = None
    self.built = True

  def call(self, inputs, states, training=None):
    if self.in_dropout_mask is None and
       self.use_dropout_mask is True:
         self.in_dropout_mask = K.dropout(
               array_ops.ones_like(inputs),
               self.in_dropout)
      
    if self.recur_dropout_mask is None and
       self.use_recur is True:
           self.recur_dropout_mask = K.dropout(
                 array_ops.ones_like(self.kern_3),
                 self.recur_dropout)

    self.connectivity_kern_1 = K.dropout(
                     array_ops.ones_like(self.kern_1),
                     self.connectivity_1)
    self.connectivity_kern_2 = K.dropout(
                     array_ops.ones_like(self.kern_2),
                     self.connectivity_2)
    self.connectivity_kern_3 = K.dropout(
                     array_ops.ones_like(self.kern_3),
                     self.connectivity_3)
                                   
    K.set_value(self.kern_1,
                spec_normalize(self.kern_1)*
                self.connectivity_kern_1)
    K.set_value(self.kern_2,
                spec_normalize(self.kern_2)*
                self.connectivity_kern_2)
    K.set_value(self.kern_3,
                spec_normalize(self.kern_3)*
                self.connectivity_kern_3)
    
    if self.use_clock is True:
       self.clock_kernel = array_ops.ones_like(self.kern_1) * clock_rate

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)

  def get_config(self):
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout
    }
    base_config = super(SimpleRNNCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

                                                                               # pylint: disable=line-too-long
"""func to get layer 2 col
    layer "2" 's size =
    total_reservoir_size - (layer_1_size plus layer_3_size) """
                                                                               # pylint: enable=line-too-long
def get_layer_dim(self, row, col, out_col=None, units):                        # pylint: disable=redefined-builtin
  if not out_col:
    out_c = col
  l_2_size = units - (row * col) - (row * out_c)    
  return (int) (l_2_size / row)
                                                                             # pylint: disable=line-too-long
 #function to ensure spectral radius is within boundaries
  """instead of reinitializing the matrix till we get the right one,
   we divide it by the greatest eigenvalue
   to make its spectral radius within desired bounds"""                    
                                                                                # pylint: enable=line-too-long
 def spec_normalize(kernel):
      if not kernel:
        raise ValueError('Kernel passed to normalize is Null')
      else:
        sing = tf.svd(kernel, compute_uv=False)
        sing_max = tf.reduce_max(sing) 
        if(sing_max > spec_radius and
           math.pow(sing_max,2) > spec_radius):                                  #if singular and eigen values greater 
              alpha = (1/sing_max+                                               #then choose any number between greatest singular value
                       1/math.pow(sing_max,2))/2                                 # and greatest eigen value ( i choose the mean)
              kernel = alpha * kernel                                            #to be the divisor
      return kernel
    
                                                                                    # pylint: disable=line-too-long    
""" alternative method for checking spectral radius but of different accuracy 
    as results tend to differ greatly some times
   
    weight_arr = K.get_value(self.kernel)  #get tensor from 'variable'
    pca = PCA(n_components=None,           #config PCA multidimensional array
              svd_solver='full')
    pca.fit(weight_arr)                          #get PCA multidimensional array

    weight_arr_sing_vector = pca.singular_values_                   #get "feature vector" which is a fancy name for singular/eigen values array     
    weight_max_sing = np.amax(abs(weight_arr_sing_vector.flatten()))           #get max abs singular value"""
                                                                                    # pylint: enable=line-too-long

def _generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype):
  if inputs is not None:
    batch_size = array_ops.shape(inputs)[0]
    dtype = inputs.dtype
  return _generate_zero_filled_state(batch_size, cell.state_size, dtype)

def _generate_zero_filled_state(batch_size_tensor, state_size, dtype):
  """Generate a zero filled tensor with shape [batch_size, state_size]."""
  if None in [batch_size_tensor, dtype]:
    raise ValueError(
        'batch_size and dtype cannot be None while constructing initial state: '
        'batch_size={}, dtype={}'.format(batch_size_tensor, dtype))
  if _is_multiple_state(state_size):
    states = []
    for dims in state_size:
      flat_dims = tensor_shape.as_shape(dims).as_list()
      init_state_size = [batch_size_tensor] + flat_dims
      init_state = array_ops.zeros(init_state_size, dtype=dtype)
      states.append(init_state)
    return states
  else:
    flat_dims = tensor_shape.as_shape(state_size).as_list()
    init_state_size = [batch_size_tensor] + flat_dims
    return array_ops.zeros(init_state_size, dtype=dtype)

def _pad(in_tensor,dims):
    S= tf.shape(in_tensor)
    padding = [[0, n-S[i]] for(i,n) in enumerate(dims)]
    return tf.pad(in_tensor, padding, 'CONSTANT', constant_value=0)

def _preprocess_format(input_shape,tf_data_format):
        tf_data_format = tf_data_format.upper()
        if tf_data_format not in {'NHWC', 'NCHW'}:
           raise ValueError('Unknown data_format: ' + str(tf_data_format))

        if tf_data_format == 'NCHW':
           if not _has_nchw_support():
              input_shape = array_ops.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
              tf_data_format = 'NHWC'
        else:
           tf_data_format = 'NCHW'

        if self.tf_data_format == 'NCHW':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        if self.tf_data_format == 'NCHW':
            r_axis, c_axis, out_c_axis = 2, 3, 4
        else:
            r_axis, c_axis, out_c_axis = 1, 2, 4
        return input_shape[r_axis], input_shape[c_axis], input_shape[out_c_axis], input_dim
   
