from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


def encode(seq, hidden_size):
  """Implement conv attentional sequence encoder."""
  with tf.variable_scope('encoder_embs'):
    src_embeddings = tf.get_variable('src_emb', [vocab_size, hidden_size])
    pos_embeddings = tf.get_variable('pos_emb', [200, hidden_size])

    seq_embs = tf.nn.embedding_lookup(src_embeddings, seq)
    pos_embs = tf.nn.embedding_lookup(pos_embeddings,
                                      tf.tile(tf.range(tf.shape(seq)[1]),
                                              [tf.shape(seq)[1], 1]))

    seqpos_embs = seq_embs + pos_embs

    # projection of previous hidden state onto source word space
    tgt_hid_proj = slim.fully_connected(prev_h, hidden_size, 'tgt_hid_proj')
    tgt_cel_proj = slim.fully_connected(prev_c, hidden_size, 'tgt_cel_proj')


