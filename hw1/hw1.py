import numpy as np
import random
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from load_data import DataGenerator
from tensorflow.python.platform import flags
from tensorflow.keras import layers

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_classes', 5,
                     'number of classes used in classification (e.g. 5-way classification).')

flags.DEFINE_integer('num_samples', 1,
                     'number of examples used for inner gradient update (K for K-shot learning).')

flags.DEFINE_integer('meta_batch_size', 16,
                     'Number of N-way classification tasks per batch')


def loss_function(preds, labels):
    """
    Computes MANN loss
    Args:
        preds: [B, K+1, N, N] network output
        labels: [B, K+1, N, N] labels
    Returns:
        scalar loss
    """
    #############################
    #### YOUR CODE GOES HERE ####

    print("labels.shape: ", labels[:, -1:].get_shape().as_list())
    loss = tf.losses.softmax_cross_entropy(labels[:, -1:], preds[:, -1:], reduction=tf.losses.Reduction.NONE)
    print(loss.eval(), "shape: ", loss.get_shape().as_list())
    loss = tf.losses.softmax_cross_entropy(labels[:, -1], preds[:, -1], reduction=tf.losses.Reduction.NONE)
    print(loss.eval(), "shape: ", loss.get_shape().as_list())
    loss = tf.reduce_mean(loss)

    return loss
    #############################


class MANN(tf.keras.Model):

    def __init__(self, num_classes, samples_per_class):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.layer1 = tf.keras.layers.LSTM(128, return_sequences=True)
        self.layer2 = tf.keras.layers.LSTM(num_classes, return_sequences=True)

    def call(self, input_images, labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        #############################
        #### YOUR CODE GOES HERE ####

        _, K_1, N, D = input_images.shape
        input_labels = tf.concat([labels[:, :-1, :, :],
                                  tf.zeros_like(labels[:, -1:])],
                                 axis=1)
        images_labels = tf.concat([input_images, input_labels], -1)
        hidden = self.layer1(tf.reshape(images_labels, [-1, K_1*N, D+N]))   # specifying B raises an error, because the dim can be Dimension(None)
        scores = self.layer2(hidden)
        out = tf.reshape(scores, [-1, K_1, N, N])

        #############################
        return out

with tf.Session() as sess:
    B, K, N = 2, 1, 2
    labels = np.array([[0, 1, 1, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 1, 1, 0]]).reshape([B, K+1, N, N])
    preds = np.array([[0.5, 0.5, 0.7, 0.3, 0.8, 0.2, 0.3, 0.7], [0.9, 0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1]]).reshape([B, K+1, N, N])
    loss = loss_function(tf.convert_to_tensor(preds), tf.convert_to_tensor(labels))
    # data_generator = DataGenerator(FLAGS.num_classes, FLAGS.num_samples + 1)
    # ims, labels = data_generator.sample_batch('train', FLAGS.meta_batch_size)
    # o = MANN(FLAGS.num_classes, FLAGS.num_samples + 1)
    # out = o(ims.astype(np.float32), labels.astype(np.float32))
    # loss = loss_function(tf.convert_to_tensor(out), tf.convert_to_tensor(labels))
exit(0)

ims = tf.placeholder(tf.float32, shape=(
    None, FLAGS.num_samples + 1, FLAGS.num_classes, 784))
labels = tf.placeholder(tf.float32, shape=(
    None, FLAGS.num_samples + 1, FLAGS.num_classes, FLAGS.num_classes))

data_generator = DataGenerator(
    FLAGS.num_classes, FLAGS.num_samples + 1)

o = MANN(FLAGS.num_classes, FLAGS.num_samples + 1)
out = o(ims, labels)

loss = loss_function(out, labels)
optim = tf.train.AdamOptimizer(0.001)
optimizer_step = optim.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    for step in range(50000):
        i, l = data_generator.sample_batch('train', FLAGS.meta_batch_size)
        feed = {ims: i.astype(np.float32), labels: l.astype(np.float32)}
        _, ls = sess.run([optimizer_step, loss], feed)

        if step % 100 == 0:
            print("*" * 5 + "Iter " + str(step) + "*" * 5)
            i, l = data_generator.sample_batch('test', 100)
            feed = {ims: i.astype(np.float32),
                    labels: l.astype(np.float32)}
            pred, tls = sess.run([out, loss], feed)
            print("Train Loss:", ls, "Test Loss:", tls)
            pred = pred.reshape(
                -1, FLAGS.num_samples + 1,
                FLAGS.num_classes, FLAGS.num_classes)
            pred = pred[:, -1, :, :].argmax(2)
            l = l[:, -1, :, :].argmax(2)
            print("Test Accuracy", (1.0 * (pred == l)).mean())
