# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Contains the TFExampleDecoder its associated helper classes.

The TFExampleDecode is a DataDecoder used to decode TensorFlow Example protos.
In order to do so each requested item must be paired with one or more Example
features that are parsed to produce the Tensor-based manifestation of the item.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from tensorflow.contrib.slim.python.slim.data import data_decoder
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.contrib.slim.python.slim.data.tfexample_decoder import ItemHandler
import tensorflow as tf


class Np_Array_Float_Tensor(ItemHandler):
  """An ItemHandler that returns a parsed Tensor."""

  def __init__(self, tensor_key, shape_keys=None, shape=None, default_value=0):
    """Initializes the Tensor handler.

    Tensors are, by default, returned without any reshaping. However, there are
    two mechanisms which allow reshaping to occur at load time. If `shape_keys`
    is provided, both the `Tensor` corresponding to `tensor_key` and
    `shape_keys` is loaded and the former `Tensor` is reshaped with the values
    of the latter. Alternatively, if a fixed `shape` is provided, the `Tensor`
    corresponding to `tensor_key` is loaded and reshape appropriately.
    If neither `shape_keys` nor `shape` are provided, the `Tensor` will be
    returned without any reshaping.

    Args:
      tensor_key: the name of the `TFExample` feature to read the tensor from.
      shape_keys: Optional name or list of names of the TF-Example feature in
        which the tensor shape is stored. If a list, then each corresponds to
        one dimension of the shape.
      shape: Optional output shape of the `Tensor`. If provided, the `Tensor` is
        reshaped accordingly.
      default_value: The value used when the `tensor_key` is not found in a
        particular `TFExample`.

    Raises:
      ValueError: if both `shape_keys` and `shape` are specified.
    """
    if shape_keys and shape is not None:
      raise ValueError('Cannot specify both shape_keys and shape parameters.')
    if shape_keys and not isinstance(shape_keys, list):
      shape_keys = [shape_keys]
    self._tensor_key = tensor_key
    self._shape_keys = shape_keys
    self._shape = shape
    self._default_value = default_value
    keys = [tensor_key]
    if shape_keys:
      keys.extend(shape_keys)
    super(Np_Array_Float_Tensor, self).__init__(keys)

  def tensors_to_item(self, keys_to_tensors):
    tensor_encode = keys_to_tensors[self._tensor_key]
    tensor = parsing_ops.decode_raw(tensor_encode, dtypes.float32)
    shape = self._shape
    if self._shape_keys:
      shape_dims = []
      for k in self._shape_keys:
        shape_dim = keys_to_tensors[k]
        if isinstance(shape_dim, sparse_tensor.SparseTensor):
          shape_dim = sparse_ops.sparse_tensor_to_dense(shape_dim)
        shape_dims.append(shape_dim)
      shape = array_ops.reshape(array_ops.stack(shape_dims), [-1])
    if isinstance(tensor, sparse_tensor.SparseTensor):
      if shape is not None:
        tensor = sparse_ops.sparse_reshape(tensor, shape)
      tensor = sparse_ops.sparse_tensor_to_dense(tensor, self._default_value)
    else:
      if shape is not None:
        '''
        tensor = array_ops.reshape(tensor, shape)
        '''         
        tensor = array_ops.reshape(tensor, [-1,shape[-1]])
        tensor_shape = tf.shape(tensor)
        #print("*************************tensor_shape :",tensor_shape)
        tenosr_zero = tf.zeros_like(tensor)
        if shape[0]!=-1:
            tensor = tf.cond(tensor_shape[0]>=shape[0], lambda : tensor[:shape[0]], \
                                               lambda : tf.concat([tensor,tenosr_zero[:(shape[0]-tensor_shape[0])]],axis=0))
                                       
    return tensor
    
    
    
    
class String_Pad_Null_Tensor(ItemHandler):
  """An ItemHandler that returns a parsed Tensor."""

  def __init__(self, tensor_key, shape_keys=None, shape=None, default_value=0):
    """Initializes the Tensor handler.

    Tensors are, by default, returned without any reshaping. However, there are
    two mechanisms which allow reshaping to occur at load time. If `shape_keys`
    is provided, both the `Tensor` corresponding to `tensor_key` and
    `shape_keys` is loaded and the former `Tensor` is reshaped with the values
    of the latter. Alternatively, if a fixed `shape` is provided, the `Tensor`
    corresponding to `tensor_key` is loaded and reshape appropriately.
    If neither `shape_keys` nor `shape` are provided, the `Tensor` will be
    returned without any reshaping.

    Args:
      tensor_key: the name of the `TFExample` feature to read the tensor from.
      shape_keys: Optional name or list of names of the TF-Example feature in
        which the tensor shape is stored. If a list, then each corresponds to
        one dimension of the shape.
      shape: Optional output shape of the `Tensor`. If provided, the `Tensor` is
        reshaped accordingly.
      default_value: The value used when the `tensor_key` is not found in a
        particular `TFExample`.

    Raises:
      ValueError: if both `shape_keys` and `shape` are specified.
    """
    if shape_keys and shape is not None:
      raise ValueError('Cannot specify both shape_keys and shape parameters.')
    if shape_keys and not isinstance(shape_keys, list):
      shape_keys = [shape_keys]
    self._tensor_key = tensor_key
    self._shape_keys = shape_keys
    self._shape = shape
    self._default_value = default_value
    keys = [tensor_key]
    if shape_keys:
      keys.extend(shape_keys)
    super(String_Pad_Null_Tensor, self).__init__(keys)

  def tensors_to_item(self, keys_to_tensors):
    tensor_encode = keys_to_tensors[self._tensor_key]
    tensor = tensor_encode
    shape = self._shape
    if self._shape_keys:
      shape_dims = []
      for k in self._shape_keys:
        shape_dim = keys_to_tensors[k]
        if isinstance(shape_dim, sparse_tensor.SparseTensor):
          shape_dim = sparse_ops.sparse_tensor_to_dense(shape_dim)
        shape_dims.append(shape_dim)
      shape = array_ops.reshape(array_ops.stack(shape_dims), [-1])
    if isinstance(tensor, sparse_tensor.SparseTensor):
      if shape is not None:
        tensor = sparse_ops.sparse_reshape(tensor, shape)
      tensor = sparse_ops.sparse_tensor_to_dense(tensor, self._default_value)
    else:
      if shape is not None:
        '''
        tensor = array_ops.reshape(tensor, shape)
        '''
        tensor = array_ops.reshape(tensor, [-1])
        tensor_shape = tf.shape(tensor)
        #print("*************************tensor_shape :",tensor_shape)
        tenosr_null = tf.constant(value='',
                                dtype=tf.string,
                                shape=[shape[0]],
                                name='string_null',
                                verify_shape=False
                               )
        if shape[0]!=-1:
            tensor = tf.cond(tensor_shape[0]>=shape[0], lambda : tensor[:shape[0]], \
                                               lambda : tf.concat([tensor,tenosr_null[:(shape[0]-tensor_shape[0])]],axis=0))
                                       
    return tensor