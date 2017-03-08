from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

def length(seq):
  length = tf.reduce_sum(tf.sign(seq), reduction_indices=1)
  return tf.cast(length, tf.int32)


def encode(src, src_vocab_size, tgt, tgt_vocab_size, hidden_size, window=5):
  """Implement conv attentional sequence encoder."""
  with tf.variable_scope('encoder_embs'):
    src_embeddings = tf.get_variable('src_emb', [src_vocab_size, hidden_size])
    pos_embeddings = tf.get_variable('pos_emb', [50, hidden_size])
    inp_embeddings = tf.get_variable('inp_emb', [tgt_vocab_size, hidden_size])
    tgt_embeddings = tf.get_variable('tgt_emb', [tgt_vocab_size, hidden_size])

    seq_embs = tf.nn.embedding_lookup(src_embeddings, seq)
    pos_embs = tf.nn.embedding_lookup(pos_embeddings,
                                      tf.tile(tf.range(tf.shape(seq)[1]),
                                              [tf.shape(seq)[1], 1]))
    # seqpos_embs of shape [batch_size, max_time, hidden_size].
    seqpos_embs = seq_embs + pos_embs

    # windowed_seqpos_embs of shape [batch_size, max_time, 1, hidden_size].
    windowed_seqpos_embs = tf.avg_pool(tf.expand_dims(seqpos_embs, 1),
                                       [1, window, 1, 1], padding='SAME')
    windowed_seqpos_embs = tf.squeeze(windowed_seqpos_embs, 2)

    tgt_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    tgt_ta = tgt_ta.unpack(tgt)
    tgt_seqlen = length(tgt)

    cell = tf.nn.rnn_cell.LSTMCell(hidden_size)

    def loop_fn(time, cell_output, cell_state, loop_state):
      emit_output = cell_output  # == None for time == 0
      if cell_output is None:  # time == 0
        next_cell_state = cell.zero_state(batch_size, tf.float32)
      else:
        next_cell_state = cell_state

      prev_h = next_cell_state.h
      prev_c = next_cell_state.c
      tgt_t = tgt_ta.read(time)

      # projection of previous hidden state onto source word space
      tgt_hid_proj = slim.fully_connected(prev_h, hidden_size, 'tgt_hid_proj')
      tgt_cel_proj = slim.fully_connected(prev_c, hidden_size, 'tgt_cel_proj')
      tgt_emb_t = tf.nn.embedding_lookup(tgt_embeddings, tgt_t)

      # tgt_rep of shape [batch_size, hidden_size].
      tgt_rep = tgt_hid_proj + tgt_cel_proj + tgt_embs
      tgt_rep = tf.expand_dims(tgt_rep, 2)

      attn_scores = tf.squeeze(tf.matmul(windowed_seqpos_embs, tgt_rep), 2)
      # attn of shape [batch_size, max_time].
      conv_attn_aux = seqpos_embs * tf.softmax(attn_scores)

      elements_finished = (time >= tgt_seqlen)
      finished = tf.reduce_all(elements_finished)
      next_input = tf.cond(
          finished,
          lambda: tf.zeros([batch_size, hidden_size], dtype=tf.float32),
          lambda: conv_attn_aux + tf.nn.embedding_lookup(inp_embeddings, tgt_t))
      next_loop_state = None
      return (elements_finished, next_input, next_cell_state,
              emit_output, next_loop_state)

    outputs_ta, final_state, _ = raw_rnn(cell, loop_fn)
    return outputs_ta.pack()




