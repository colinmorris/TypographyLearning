import tensorflow as tf
import dataset as ds
import numpy as np
from common import INPUT_SIZE, IMG_SIZE, HEADER
import common

"""TODO:
-Make a makefile
  -Clean up output dir between runs, or write to separate subdirs
  -Run imagemagick to generate scaled-up versions
-Clean up all this cruft
-Do globals the tf way
-Experiment with testing on letters at previously unseen positions (intentionally generate some wacky random strings and reserve for test set)
- Add image examples to TensorBoard
- Visualize the activations. That'd be super cool and insightful.
- When we get a 'cache miss' on a particular letter, why does it make the whole image fuzzier?
  - Well the 'second letter is j' input node hasn't had its weights ever updated, so they'll just be set to whatever they started at - which was gaussian noise!
"""

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                                        'Must divide evenly into the dataset sizes.')

flags.DEFINE_string('train_dir', 'data/imgs', 'Directory to put the training data.')
flags.DEFINE_string('test_dir', None, 'Directory to put the test data.')
flags.DEFINE_integer('iterations', 2000, 'How many training iterations to run.')

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


if FLAGS.test_dir is None:
  # We just have one data dir, so randomly split it into train/test data
  train, test = ds.read_traintest_data(FLAGS.train_dir)
else:
  train = ds.dataset_from_dir(FLAGS.train_dir)
  test = ds.dataset_from_dir(FLAGS.test_dir)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])
y_ = tf.placeholder(tf.float32, shape=[None, IMG_SIZE])

# Initialization makes a pretty significant difference for unseen character-positions.
# Zero init seems to give much more pleasing results.
W = tf.Variable(tf.zeros([INPUT_SIZE,IMG_SIZE]))
#W = tf.Variable(tf.truncated_normal([INPUT_SIZE,IMG_SIZE], stddev=0.1))
variable_summaries(W, 'weights')
b = tf.Variable(tf.zeros([IMG_SIZE]))
tf.histogram_summary('biases', b)

train_writer = tf.train.SummaryWriter(SUMMARIES_DIR + '/train', sess.graph, max_queue=10)
test_writer = tf.train.SummaryWriter(SUMMARIES_DIR + '/test', sess.graph)

# The predicted image
y = tf.matmul(x,W) + b
variable_summaries(y, 'pixel_weights')

# This is dumb, but whatever. Squared error seemed to make things crazy.
loss = tf.reduce_sum(abs(y - y_))
# TODO: Adam works great here, and Gradient Descent is horrible. Bad learning rate or what?
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
train_step = tf.train.AdamOptimizer().minimize(loss)

# Mean absolute difference per pixel (scaled up to 255)
avg_loss = tf.reduce_mean(255 * abs(y-y_))
tf.scalar_summary('loss', avg_loss)

# Save some images
def reshape(img):
  return tf.reshape(
    tf.slice(img, begin=[0, 0], size=[1, -1]), # Just take the first image
    # -1 means "make this whatever it needs to be to fit"
    [-1, common.IMG_HEIGHT, common.IMG_WIDTH, 1])

merged = tf.merge_all_summaries()
tf.initialize_all_variables().run()

for i in range(FLAGS.iterations):
  batch = train.next_batch(FLAGS.batch_size)
  summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1]})

  if i > 0 and (((i % 50) == 0 and i < 500) or (i % 100 == 0)):
    # Need to move this inside because adds seemed like they were getting randomly dropped

    train_writer.add_summary(summary, i)

    # TODO: The fact that I need to make a new one every time almost feels like a bug
    orig_img = tf.image_summary("{:09d}".format(i), reshape(y_), max_images=5)
    output_img = tf.image_summary("{:09d}".format(i), reshape(y), max_images=5)

    # TODO: I don't really like how tf pixelizes my floats. Should rework this to
    # match the behaviour of write_img_data fn
    orig_sum = orig_img.eval(feed_dict={x: batch[0], y_: batch[1]})
    output_sum = output_img.eval(feed_dict={x: batch[0], y_: batch[1]})

    # This is a silly hack so that the originals and outputs show up side-by-side in tensorboard
    test_writer.add_summary(orig_sum)
    train_writer.add_summary(output_sum)
    train_writer.flush()
    test_writer.flush()
    # TODO: Also write examples from the test set?
    print "Training loss: {}".format(avg_loss.eval(feed_dict={x: batch[0], y_: batch[1]}))
    # # Evaluate accuracy and save summary
    # summary, _ = sess.run([merged, avg_loss], feed_dict={x: batch[0], y_: batch[1]})
    # train_writer.add_summary(summary, i)
    #
    # summary, _ = sess.run([merged, avg_loss], feed_dict={x: test.texts, y_:test.images})
    # test_writer.add_summary(summary, i)

# Bleh hacks
#train_writer.add_summary(combined_summary)

  #train_step.run(feed_dict={x: batch[0], y_: batch[1]})

print "Avg loss per pixel on test set (0-255): {}".format(avg_loss.eval(feed_dict={x: test.texts, y_:test.images}))
for text, label in zip(test.texts, test.text_labels):
  img_data = y.eval(feed_dict={x: np.reshape(text, [1, len(text)])})
  f = open('outputs/' + label.replace(' ', '_') + '.pgm', 'wb')
  write_img_data(f, img_data)
  f.close()

