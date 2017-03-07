from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import makedata as md
import numpy as np


def get_example(keys, fname, num_epochs=None):
  filename_queue = tf.train.string_input_producer([fname],
                                                  num_epochs=num_epochs)
  reader = tf.TFRecordReader()

  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    features={
      'src': tf.FixedLenFeature([], tf.string),
      'src_ids': tf.VarLenFeature(tf.int64),
      'target': tf.FixedLenFeature([], tf.string),
      'target_ids': tf.VarLenFeature(tf.int64),
    }
  )
  return [features[k] for k in keys]

def words_to_ids_fn(dict_, bptt):
  def words_to_ids(batch_lines):
    batch = []
    maxlen = 0
    for line in batch_lines:
      ids = map(lambda w: dict_[w] if w in dict_ else dict_[md.UNK],
                line.split())
      batch.append(ids)
      maxlen = max(maxlen, len(ids))
    res = np.zeros([len(batch), min(bptt, maxlen)], dtype=np.int64)

    for i, line in enumerate(batch):
      length = min(bptt, len(line))
      res[i, :length] = line[:length]
    return res
  return words_to_ids
