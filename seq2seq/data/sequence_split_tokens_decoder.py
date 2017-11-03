# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A decoder for tf.SequenceExample"""

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data import data_decoder


class TFSEquenceSplitTokensDecoder(data_decoder.DataDecoder):
  """A decoder for TensorFlow Examples.
  Decoding Example proto buffers is comprised of two stages: (1) Example parsing
  ,and (2) tensor manipulation.
  In the first stage, the tf.parse_example function is called with a list of
  FixedLenFeatures and SparseLenFeatures. These instances tell TF how to parse
  the example. The output of this stage is a set of tensors. 
  And in this stage, the decoder will add  "START" and "END" for tokens
  In the second stage, the resulting tensors are manipulated to provide the
  requested 'item' tensors.
  To perform this decoding operation, an ExampleDecoder is given a list of
  ItemHandlers. Each ItemHandler indicates the set of features for stage 1 and
  contains the instructions for post_processing its tensors for stage 2.
  """

  def __init__(self, context_keys_to_features, sequence_keys_to_features,
               items_to_handlers,
               delimiter=" ",
               tokens_feature_name="tokens",
               length_feature_name="length",
               prepend_token=None,
               append_token=None):
    """Constructs the decoder.
    Args:
      keys_to_features: a dictionary from TF-Example keys to either
        tf.VarLenFeature or tf.FixedLenFeature instances. See tensorflow's
        parsing_ops.py.
      items_to_handlers: a dictionary from items (strings) to ItemHandler
        instances. Note that the ItemHandler's are provided the keys that they
        use to return the final item Tensors.
    """
    self.delimiter = delimiter
    self.tokens_feature_name = tokens_feature_name
    self.length_feature_name = length_feature_name
    self.prepend_token = prepend_token
    self.append_token = append_token

    self._context_keys_to_features = context_keys_to_features
    self._sequence_keys_to_features = sequence_keys_to_features
    self._items_to_handlers = items_to_handlers

  def list_items(self):
    """See base class."""
    return list(self._items_to_handlers.keys())

  def decode(self, serialized_example, items=None):
    """Decodes the given serialized TF-example.
    Args:
      serialized_example: a serialized TF-example tensor.
      items: the list of items to decode. These must be a subset of the item
        keys in self._items_to_handlers. If `items` is left as None, then all
        of the items in self._items_to_handlers are decoded.
    Returns:
      the decoded items, a list of tensor.
    """
    context, sequence = tf.parse_single_sequence_example(
        serialized_example, self._context_keys_to_features,
        self._sequence_keys_to_features)
    tokens_raw = sequence[self.tokens_feature_name]
    
    tokens = tf.string_split(tokens_raw, delimiter=self.delimiter).values

    # Optionally prepend a special token
    if self.prepend_token is not None:
      tokens = tf.concat([[self.prepend_token], tokens], 0)

    # Optionally append a special token
    if self.append_token is not None:
      tokens = tf.concat([tokens, [self.append_token]], 0)
    sequence[self.tokens_feature_name] = tokens
    # Merge context and sequence features
    example = {}
    example.update(context)
    example.update(sequence)

    all_features = {}
    all_features.update(self._context_keys_to_features)
    all_features.update(self._sequence_keys_to_features)

    # Reshape non-sparse elements just once:
    for k, value in all_features.items():
      if isinstance(value, tf.FixedLenFeature):
        example[k] = tf.reshape(example[k], value.shape)

    if not items:
      items = self._items_to_handlers.keys()

    outputs = []
    for item in items:
      handler = self._items_to_handlers[item]
      keys_to_tensors = {key: example[key] for key in handler.keys}
      outputs.append(handler.tensors_to_item(keys_to_tensors))
    return outputs
