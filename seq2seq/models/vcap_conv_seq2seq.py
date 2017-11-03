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
"""
Definition of a basic seq2seq model
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from seq2seq import graph_utils
from pydoc import locate
import tensorflow as tf
from seq2seq.contrib.seq2seq import helper as tf_decode_helper
from seq2seq.data import vocab
from seq2seq.models.seq2seq_model import Seq2SeqModel
from seq2seq.graph_utils import templatemethod
from seq2seq.models import bridges
from seq2seq.inference import beam_search
from seq2seq.encoders.conv_encoder_utils import linear_mapping_weightnorm
class VCAP_ConvSeq2Seq(Seq2SeqModel):
  """Basic Sequence2Sequence model with a unidirectional encoder and decoder.
  The last encoder state is used to initialize the decoder and thus both
  must share the same type of RNN cell.

  Args:
    source_vocab_info: An instance of `VocabInfo`
      for the source vocabulary
    target_vocab_info: An instance of `VocabInfo`
      for the target vocabulary
    params: A dictionary of hyperparameters
  """

  def __init__(self, params, mode, name="vcap_conv_seq2seq"):
    super(VCAP_ConvSeq2Seq, self).__init__(params, mode, name)
    self.encoder_class = locate(self.params["encoder.class"])
    self.decoder_class = locate(self.params["decoder.class"])

  @staticmethod
  def default_params():
    params = Seq2SeqModel.default_params().copy()
    params.update({
        "encoder.class": "seq2seq.encoders.ConvEncoderFairseq",
        "encoder.params": {},  # Arbitrary parameters for the encoder
        "decoder.class": "seq2seq.decoders.ConvDecoder",
        "decoder.params": {},  # Arbitrary parameters for the decoder
        "source.max_seq_len": 50,
        "source.reverse": False,
        "feature.dim":2048,
        "feature.shape":[30,2048],
        "target.max_seq_len": 50,
        "embedding.dim": 256,
        "embedding.init_scale": 0.04,
        "embedding.share": False,
        "position_embeddings.num_positions": 100,
        "inference.beam_search.beam_width": 0,
        "inference.beam_search.length_penalty_weight": 1.0,
        "inference.beam_search.choose_successors_fn": "choose_top_k",
        "vocab_source": "",
        "vocab_target": "", 
        "optimizer.name": "Momentum",
        "optimizer.learning_rate": 0.25,
        "optimizer.params": {"momentum": 0.99, "use_nesterov": True}, # Arbitrary parameters for the optimizer
        #"optimizer.params": { "epsilon": 0.0000008}, # Arbitrary parameters for the optimizer
        "optimizer.lr_decay_type": "exponential_decay",
        "optimizer.lr_decay_steps": 5000,  # one epoch steps
        "optimizer.lr_decay_rate": 0.9,  
        "optimizer.lr_start_decay_at": 0,  # start annealing epoch 0
        "optimizer.lr_stop_decay_at": tf.int32.max,
        "optimizer.lr_min_learning_rate": 1e-5,
        "optimizer.lr_staircase": True,
        "optimizer.clip_gradients": 0.1,
        "optimizer.clip_embed_gradients": 5,
        "optimizer.sync_replicas": 0,
        "optimizer.sync_replicas_to_aggregate": 0,
        
})
    return params
  '''      '''

  def _preprocess(self, features, labels):
    """Model-specific preprocessing for features and labels:
    In decoder pahse, the labels have add "END" and "START"
    - Creates vocabulary lookup tables for target vocab
    - Converts tokens into vocabulary ids
    """

    # Create vocabulary look for target
    target_vocab_to_id, target_id_to_vocab, target_word_to_count, _ = \
      vocab.create_vocabulary_lookup_table(self.target_vocab_info.path)

    # Add vocab tables to graph colection so that we can access them in
    # other places.
    graph_utils.add_dict_to_collection({
        "target_vocab_to_id": target_vocab_to_id,
        "target_id_to_vocab": target_id_to_vocab,
        "target_word_to_count": target_word_to_count
    }, "vocab_tables")

    if labels is None:
      return features, None

    labels = labels.copy()
    # Slices targets to max length
    if self.params["target.max_seq_len"] is not None:
      labels["target_tokens"] = labels["target_tokens"][:, :self.params[
          "target.max_seq_len"]]
      labels["target_len"] = tf.minimum(labels["target_len"],
                                        self.params["target.max_seq_len"])
    # Look up the target ids in the vocabulary
    labels["target_ids"] = target_vocab_to_id.lookup(labels["target_tokens"])

    labels["target_len"] = tf.to_int32(labels["target_len"])
    tf.summary.histogram("target_len", tf.to_float(labels["target_len"]))

    # Add to graph collection for later use
    graph_utils.add_dict_to_collection(features, "features")
    if labels:
      graph_utils.add_dict_to_collection(labels, "labels")

    return features, labels

    
  def source_embedding_fairseq(self):
    """Returns the embedding used for the source sequence.
    """
    return tf.get_variable(
        name="W",
        shape=[self.params["feature.dim"], self.params["embedding.dim"]],
        initializer=tf.random_normal_initializer(
            mean=0.0,
            stddev=0.1))

  def target_embedding_fairseq(self):
    """Returns the embedding used for the target sequence.
    """
    if self.params["embedding.share"]:
      return self.source_embedding_fairseq()
    return tf.get_variable(
        name="W",
        shape=[self.target_vocab_info.total_size, self.params["embedding.dim"]],
        initializer=tf.random_normal_initializer(
            mean=0.0,
            stddev=0.1))

  def source_pos_embedding_fairseq(self):
    return tf.get_variable(
        name="pos",
        shape=[self.params["position_embeddings.num_positions"], self.params["embedding.dim"]],
        initializer=tf.random_normal_initializer(
            mean=0.0,
            stddev=0.1))
    
  def target_pos_embedding_fairseq(self):
    return tf.get_variable(
        name="pos",
        shape=[self.params["position_embeddings.num_positions"], self.params["embedding.dim"]],
        initializer=tf.random_normal_initializer(
            mean=0.0,
            stddev=0.1))

  def _create_decoder(self, encoder_output, features, _labels):

    config = beam_search.BeamSearchConfig(
        beam_width=self.params["inference.beam_search.beam_width"],
        vocab_size=self.target_vocab_info.total_size,
        eos_token=self.target_vocab_info.special_vocab.SEQUENCE_END,
        length_penalty_weight=self.params[
            "inference.beam_search.length_penalty_weight"],
        choose_successors_fn=getattr(
            beam_search,
            self.params["inference.beam_search.choose_successors_fn"]))
    
    return self.decoder_class(
        params=self.params["decoder.params"],
        mode=self.mode,
        vocab_size=self.target_vocab_info.total_size,
        config=config,
        target_embedding=self.target_embedding_fairseq(),
        pos_embedding=self.target_pos_embedding_fairseq(),
        start_tokens=self.target_vocab_info.special_vocab.SEQUENCE_END)

  def _decode_train(self, decoder, _encoder_output, _features, labels):
    """Runs decoding in training mode"""
    target_embedded = tf.nn.embedding_lookup(decoder.target_embedding,
                                             labels["target_ids"])

    return decoder(_encoder_output, labels=target_embedded[:,:-1], sequence_length=labels["target_len"]-1)

  def _decode_infer(self, decoder, _encoder_output, features, labels):
    """Runs decoding in inference mode"""

    return decoder(_encoder_output, labels)

  @templatemethod("encode")
  def encode(self, features, labels):
    # For video captioning, the model directly encoder the features from deep cnn with pos embedding.
    source_embedded = linear_mapping_weightnorm(features['video'],self.params["embedding.dim"],\
                                                    var_scope_name="video_feat_mapping")      
    
    encoder_fn = self.encoder_class(self.params["encoder.params"], self.mode, None)
    return encoder_fn(source_embedded, self.params["feature.shape"][0])

  @templatemethod("decode")
  def decode(self, encoder_output, features, labels):
    
    decoder = self._create_decoder(encoder_output, features, labels)
     
    if self.mode == tf.contrib.learn.ModeKeys.INFER:
      return self._decode_infer(decoder, encoder_output, features,
                                labels)
    else:
      return self._decode_train(decoder, encoder_output, features,
                                labels)
