import os
import numpy as np
import random
import common
from common import INPUT_SIZE, IMG_SIZE, HEADER, TEXT_LENGTH, ALPHABET_SIZE

DEFAULT_DATA_DIR = os.path.join(common.ROOT_DIR, 'data', 'imgs')

TEST_RATIO = 0.04

class DataSet(object):

  def __init__(self, images, texts, labels=None):
    assert images.shape[0] == texts.shape[0], (
        'images.shape: %s labels.shape: %s' % (images.shape,
                                               texts.shape))
    self._num_examples = images.shape[0]

    # Convert from [0, 255] -> [0.0, 1.0].
    images = images.astype(np.float32)
    images = np.multiply(images, 1.0 / 255.0)
    self._images = images
    self._texts = texts
    self.text_labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def texts(self):
    return self._texts

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._texts = self._texts[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._texts[start:end], self._images[start:end]


def vectorize_text(t):
  """Return a one-hot encoded vector for the given input text
  """
  # Pad with spaces
  padlen = TEXT_LENGTH - len(t)
  t = t + (' ' * padlen)
  vec = np.zeros([INPUT_SIZE])
  for i, char in enumerate(t):
    # a = 0, z = 25, ' ' = 26
    if char == ' ':
      charindex = 26
    else:
      charindex = ord(char) - ord('a')
    vec[i*ALPHABET_SIZE + charindex] = 1
  return vec

def vectorize_texts(texts):
  vecs = np.zeros([len(texts), INPUT_SIZE])
  for i,text in enumerate(texts):
    vecs[i] = vectorize_text(text)
  return vecs

def read_data_sets(datadir=None):
  if datadir is None:
    datadir = DEFAULT_DATA_DIR

  texts = []
  img_files = os.listdir(datadir)
  random.seed(1337)
  # Shuffle so we get a random train/test split
  random.shuffle(img_files)
  n = len(img_files)
  img_data = np.zeros([n, IMG_SIZE], dtype=np.float32) # TODO
  for i, fname in enumerate(img_files):
    # TODO: Refactor string sanitization
    assert fname.endswith('.pgm')
    text = fname.split('.')[0].replace('_', ' ')
    texts.append(text)

    f = open(os.path.join(datadir, fname), 'rb')
    # 14 bytes of metadata
    header = f.read(14)

    assert header == HEADER, "Expected <{}> but got <{}>".format(HEADER, header)
    # 1 byte per pixel. yay.
    buf = f.read()
    data = np.frombuffer(buf, dtype=np.uint8)
    # we've got an array where all elements are 0 (black) or 255 (white)
    img_data[i] = data

  text_data = vectorize_texts(texts)

  test_idx = int(n*TEST_RATIO)

  test_set = DataSet(img_data[:test_idx], text_data[:test_idx],
                     labels=texts[:test_idx])
  train_set = DataSet(img_data[test_idx:], text_data[test_idx:])

  return train_set, test_set