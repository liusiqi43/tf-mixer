from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import reader as r
import os
import pickle as pkl

flags = tf.flags
flags.DEFINE_string('data_dir', 'data', 'Directory to read data from.')
flags.DEFINE_float('lr', 0.2, 'Learning rate.')
flags.DEFINE_float('max_gradient_norm', 10, 'Max gradient norm for generator.')
flags.DEFINE_integer('batch_size', 32, 'Batch size.')
flags.DEFINE_integer('hidden_size', 256, 'Number of hidden units.')
flags.DEFINE_integer('bptt', 25, 'Number of backprop steps through time.')
flags.DEFINE_integer('delta_steps', 3,
                     'Increment of number of steps using REINFORCE at next round.')
flags.DEFINE_integer('delta_epochs', 5,
                     'Number of epochs of each stage of REINFORCE training.')
flags.DEFINE_integer('epoch_xent', 25,
                     'Number of epochs we do with pure XENT to initialize the model.')
flags.DEFINE_string('reward', 'bleu',
                    'Reward metric used for REINFORCE. bleu|rouge.')
flags.DEFINE_string('split', 'train', 'Data split to use to run the model.')

FLAGS = flags.FLAGS


def main(unused_args):
  fname = os.path.join(FLAGS.data_dir, '%s.de-en.tfrecords' % FLAGS.split)
  src_dict = pkl.load(
    open(os.path.join(FLAGS.data_dir, 'dict.de-en.de.pkl'), 'rb'))
  target_dict = pkl.load(
    open(os.path.join(FLAGS.data_dir, 'dict.de-en.en.pkl'), 'rb'))
  examples = r.get_example(['src', 'target'], fname)

  src, target = tf.train.shuffle_batch(
    examples, FLAGS.batch_size, num_threads=4, capacity=50 * FLAGS.batch_size,
    min_after_dequeue=30 * FLAGS.batch_size)
  src_ids = tf.py_func(r.words_to_ids_fn(src_dict, FLAGS.bptt),
                       [src], [tf.int64])
  target_ids = tf.py_func(r.words_to_ids_fn(target_dict, FLAGS.bptt),
                          [target], [tf.int64])

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for _ in xrange(5):
      print(sess.run([src, src_ids, target, target_ids]))

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
  tf.app.run()
