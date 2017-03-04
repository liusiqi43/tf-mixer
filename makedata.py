
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from itertools import izip

import tensorflow as tf
import re
import os
import pickle as pkl

flags = tf.flags
flags.DEFINE_string('dst_dir', 'data', 'Directory to write to.')
flags.DEFINE_string('src_dir', 'prep', 'Directory containing preprocessed data.')
flags.DEFINE_integer('threshold', 3, 'Remove tokens appeared less than `threshold` number of times.')

FLAGS = flags.FLAGS

PAD = '<pad>'
UNK = '<unk>'
EOS = '</s>'

def cleanup_sentence(s):
  s = re.sub(r'\t', '', s)
  # remove leading and following white spaces
  s = s.strip()
  # convert multiple spaces into a single space: this is needed to
  # make the following pl.utils.split() function return only words
  # and not white spaces
  s = re.sub(r'%s+', ' ', s)
  return s

def build_dictionary(filename, threshold):
  token_to_freq = defaultdict(int)
  print('[ Reading from', filename, ']')

  with open(filename, 'r') as f:
    for line in f:
      words = cleanup_sentence(line).split(' ')

      for w in words:
        token_to_freq[w] += 1

  vocab_list = [PAD, UNK, EOS]

  for token, freq in token_to_freq.iteritems():
    if freq >= threshold:
      vocab_list.append(token)

  print('[ Done making the dictionary. ]')
  print('Training corpus statistics')
  print('Unique words:', len(token_to_freq))
  print('Total words', sum(token_to_freq.values()))
  print('[ There are effectively', len(vocab_list), 'words in the corpus. ]')

  dictionary = {w : i for i, w in enumerate(vocab_list)}
  return dictionary

def words_to_ids(words, dictionary, marking):
  ids = []
  unk_count = 0
  for w in words:
    if w not in dictionary:
      ids.append(dictionary[UNK])
      unk_count += 1
    else:
      ids.append(dictionary[w])

  if marking == 'append':
    ids.append(dictionary[EOS])
  elif marking == 'prepend':
    ids.insert(0, dictionary[EOS])
  else:
    raise RuntimeError('Unrecognized marking strategy: %s' % marking)
  return ids, unk_count, len(ids)

def build_tfrecords(src_dict, src_fname, target_dict, target_fname,
                    output_fname):
  src_unk_count = 0
  src_tokens_count = 0
  target_unk_count = 0
  target_tokens_count = 0
  lines_count = 0

  writer = tf.python_io.TFRecordWriter(output_fname)
  with open(src_fname, 'r') as fsrc, open(target_fname, 'r') as ftarget:
    for src, target in izip(fsrc, ftarget):
      lines_count += 1

      src = cleanup_sentence(src)
      target = cleanup_sentence(target)

      src_ids, unk, tok = words_to_ids(src.split(), src_dict, 'append')
      src_unk_count += unk
      src_tokens_count += tok

      target_ids, unk, tok = words_to_ids(target.split(), target_dict, 'prepend')
      target_unk_count += unk
      target_tokens_count += tok

      example = tf.train.Example(
        features=tf.train.Features(
          feature={
            'src': tf.train.Feature(
              bytes_list=tf.train.BytesList(value=[src])),
            'target': tf.train.Feature(
              bytes_list=tf.train.BytesList(value=[target])),
            'src_ids': tf.train.Feature(
              int64_list=tf.train.Int64List(value=src_ids)),
            'target_ids': tf.train.Feature(
              int64_list=tf.train.Int64List(value=src_ids)),
          }
        )
      )

      serialized = example.SerializeToString()
      writer.write(serialized)
  print('-- %s stats:' % output_fname)
  print('nlines: %d, ntokens (src: %d, tgt: %d); UNK (src: %.2f%%, tgt: %.2f%%)'
        % (lines_count, src_tokens_count, target_tokens_count,
           100 * src_unk_count / src_tokens_count,
           100 * target_unk_count / target_tokens_count))

def main(unused_args):
  if not os.path.exists(FLAGS.dst_dir):
    os.makedirs(FLAGS.dst_dir)

  datasets = {
    'train': ('train.de-en.de', 'train.de-en.en'),
    'valid': ('valid.de-en.de', 'valid.de-en.en'),
    'test': ('test.de-en.de', 'test.de-en.en')
  }

  src_dict = build_dictionary(
    os.path.join(FLAGS.src_dir, datasets['train'][0]), FLAGS.threshold)
  target_dict = build_dictionary(
    os.path.join(FLAGS.src_dir, datasets['train'][1]), FLAGS.threshold)

  for split, fnames in datasets.iteritems():
    build_tfrecords(
      src_dict, os.path.join(FLAGS.src_dir, fnames[0]),
      target_dict, os.path.join(FLAGS.src_dir, fnames[1]),
      os.path.join(FLAGS.dst_dir, '%s.de-en.tfrecords' % split))

  with open(os.path.join(FLAGS.dst_dir, 'dict.de-en.de.pkl'), 'wb') as f:
    pkl.dump(src_dict, f)

  with open(os.path.join(FLAGS.dst_dir, 'dict.de-en.en.pkl'), 'wb') as f:
    pkl.dump(target_dict, f)



if __name__ == "__main__":
  tf.app.run()







