import tensorflow as tf
import dataset as ds
import numpy as np
from common import INPUT_SIZE, IMG_SIZE, HEADER

"""TODO:
-Make a makefile
  -Clean up output dir between runs, or write to separate subdirs
  -Run imagemagick to generate scaled-up versions
-Clean up all this cruft
-Do globals the tf way
-Experiment with testing on letters at previously unseen positions (intentionally generate some wacky random strings and reserve for test set)
- Add image examples to TensorBoard
"""

BATCH_SIZE = 50

SUMMARIES_DIR = 'summaries'

# Copied from https://www.tensorflow.org/versions/r0.8/how_tos/summaries_and_tensorboard/index.html
def variable_summaries(var, name):
  with tf.name_scope("summaries"):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.scalar_summary('sttdev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)

def write_img_data(fp, data):
  """Given a prediction, draw it at the given file location"""
  fp.write(HEADER)
  # Data will be (probably) in the range (0, 1). Need to get back to (0, 255)
  grey_values = np.clip(data * 255, 0, 255).astype(np.uint8)
  fp.write(np.getbuffer(grey_values))


train, test = ds.read_data_sets()

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])
y_ = tf.placeholder(tf.float32, shape=[None, IMG_SIZE])

# TODO: Not sure if initialization makes a difference
#W = tf.Variable(tf.zeros([INPUT_SIZE,IMG_SIZE]))
W = tf.Variable(tf.truncated_normal([INPUT_SIZE,IMG_SIZE], stddev=0.1))
variable_summaries(W, 'weights')
b = tf.Variable(tf.zeros([IMG_SIZE]))
tf.histogram_summary('biases', b)

merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter(SUMMARIES_DIR + '/train', sess.graph)

# The predicted image
y = tf.matmul(x,W) + b

# This is dumb, but whatever. Squared error seemed to make things crazy.
loss = tf.reduce_sum(abs(y - y_))
# TODO: Adam works great here, and Gradient Descent is horrible. Bad learning rate or what?
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
train_step = tf.train.AdamOptimizer().minimize(loss)

tf.initialize_all_variables().run()

for i in range(1000):
  batch = train.next_batch(BATCH_SIZE)
  summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1]})
  train_writer.add_summary(summary, i)

  #train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# TODO: Quantitative accuracy eval


for text, label in zip(test.texts, test.text_labels):
  img_data = y.eval(feed_dict={x: np.reshape(text, [1, len(text)])})
  f = open('outputs/' + label.replace(' ', '_') + '.pgm', 'wb')
  write_img_data(f, img_data)
  f.close()

train_writer.flush()
