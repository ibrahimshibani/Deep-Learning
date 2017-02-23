# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple, end-to-end, LeNet-5-like convolutional rgbd_10 model example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import sys
import time
import math

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import cPickle as pickle
#from sklearn.metrics import confusion_matrix

import input_data

# DONE
# These are some useful constants that you can use in your code.
# Feel free to ignore them or change them. 
IMAGE_SIZE = 32
NUM_LABELS = 10
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 30 
EVAL_BATCH_SIZE = 1024
EVAL_FREQUENCY = 100  # Number of steps between evaluations.
# This is where the data gets stored
#TRAIN_DIR = 'data'
# HINT:
# if you are working on the computers in the pool and do not want
# to download all the data you can use the pre-loaded data like this:
TRAIN_DIR = '/home/mllect/data/rgbd'


def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  return tf.float32

def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      np.sum(np.argmax(predictions, 1) == labels) /
      predictions.shape[0])

def fake_data(num_images, channels):
  """Generate a fake dataset that matches the dimensions of rgbd_10 dataset."""
  data = np.ndarray(
      shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, channels),
      dtype=np.float32)
  labels = np.zeros(shape=(num_images,), dtype=np.int64)
  for image in xrange(num_images):
    label = image % 2
    data[image, :, :, 0] = label - 0.5
    labels[image] = label
  return data, labels

def main(argv=None):  # pylint: disable=unused-argument
  if FLAGS.self_test:
    print('Running self-test.')
    NUM_CHANNELS = 1
    train_data, train_labels = fake_data(256, NUM_CHANNELS)
    validation_data, validation_labels = fake_data(EVAL_BATCH_SIZE, NUM_CHANNELS)
    test_data, test_labels = fake_data(EVAL_BATCH_SIZE, NUM_CHANNELS)
    num_epochs = 1
  else:
    if (FLAGS.use_rgbd):
      NUM_CHANNELS = 4
      print('****** RGBD_10 dataset ******') 
      print('* Input: RGB-D              *')
      print('* Channels: 4               *') 
      print('*****************************')
    else:
      NUM_CHANNELS = 3
      print('****** RGBD_10 dataset ******') 
      print('* Input: RGB                *')
      print('* Channels: 3               *') 
      print('*****************************')
    # Load input data
    data_sets = input_data.read_data_sets(TRAIN_DIR, FLAGS.use_rgbd)
    num_epochs = NUM_EPOCHS

    train_data = data_sets.train.images
    train_labels= data_sets.train.labels
    test_data = data_sets.test.images
    test_labels = data_sets.test.labels 
    validation_data = data_sets.validation.images
    validation_labels = data_sets.validation.labels

  train_size = train_labels.shape[0]

  # DONE:
  # After this you should define your network and train it.
  # Below you find some starting hints. For more info have
  # a look at online tutorials for tensorflow:
  # https://www.tensorflow.org/versions/r0.11/tutorials/index.html
  # Your goal for the exercise will be to train the best network you can
  # please describe why you did chose the network architecture etc. in
  # the one page report, and include some graph / table showing the performance
  # of different network architectures you tried.
  #
  # Your end result should be for RGB-D classification, however, you can
  # also load the dataset with NUM_CHANNELS=3 to only get an RGB version.
  # A good additional experiment to run would be to cmompare how much
  # you can gain by adding the depth channel (how much better the classifier can get)
  
  # This is where training samples and labels are fed to the graph.
  # These placeholder nodes will be fed a batch of training data at each
  # training step using the {feed_dict} argument to the Run() call below.
  train_data_node = tf.placeholder(
      data_type(),
      shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
  eval_data = tf.placeholder(
     data_type(),
     shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  eval_labels = tf.placeholder(tf.int64, shape=(EVAL_BATCH_SIZE,))

  
  # Done: define your model here
  # Define Weight creation function
  def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, seed=SEED, dtype=data_type())
    return tf.Variable(initial)
  
  # Define Bias creation function
  def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=data_type())
    return tf.Variable(initial)

  # Define Convolutional layer creation function
  def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  # Define Convolutional layer creation function 
  def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

  ## Define the first convolutional neural network layer
  CONV_LAYER1_FEAT = 32
  W1 = weight_variable([6, 6, NUM_CHANNELS, CONV_LAYER1_FEAT])
  b1 = bias_variable([CONV_LAYER1_FEAT])
  ## Define the second convolutional neural network layer
  CONV_LAYER2_FEAT = 64
  W2 = weight_variable([6, 6, CONV_LAYER1_FEAT, CONV_LAYER2_FEAT])
  b2 = bias_variable([CONV_LAYER2_FEAT])

  W_fc1 = weight_variable([int(math.pow(IMAGE_SIZE//4, 2) * CONV_LAYER2_FEAT), 1024])
  b_fc1 = bias_variable([1024])
  W_fc2 = weight_variable([1024, NUM_LABELS])
  b_fc2 = bias_variable([NUM_LABELS])
  
  # Define model to be used for train and validation
  def model(data):
    h_conv1 = tf.nn.relu(conv2d(data, W1) + b1)    
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W2) + b2)    
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, int(math.pow(IMAGE_SIZE//4, 2)*CONV_LAYER2_FEAT)])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    ## Define Output Layer
    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    return y_conv
      # Done: compute the loss of the model here
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model(train_data_node), tf.one_hot(train_labels_node, NUM_LABELS, on_value = 1, off_value = 0)))  
      # Done
      # then create an optimizer to train the model
      # HINT: you can use the various optimizers implemented in TensorFlow.
      #       For example, google for: tf.train.AdamOptimizer()
  train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
  train_softmax = tf.nn.softmax(model(train_data_node))
  validation_softmax = tf.nn.softmax(model(eval_data))
  correct_prediction = tf.equal(tf.argmax(train_softmax,1), train_labels_node)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  error = 1 - accuracy
  eval_correct_prediction = tf.equal(tf.argmax(validation_softmax,1), eval_labels)
  eval_accuracy  = tf.reduce_mean(tf.cast(eval_correct_prediction, tf.float32))
  eval_error = 1 - eval_accuracy
  
  # Function to evaluate error of batches
  def eval_batches(data ,labels):
    size = data.shape[0]
    error = 0
    for step in xrange(int(math.ceil(size/EVAL_BATCH_SIZE))):
      lowerLimit = ((step) * EVAL_BATCH_SIZE)
      upperLimit = ((step+1) * EVAL_BATCH_SIZE)
      if(upperLimit>size):
        upperLimit = size
        lowerLimit = size - EVAL_BATCH_SIZE
      batch_data = data[lowerLimit:upperLimit,]
      batch_label = labels[lowerLimit:upperLimit]
      validation_error = eval_error.eval(feed_dict={eval_data: batch_data, eval_labels: batch_label})
      error = error + validation_error
    return error/(step+1)

  # DONE:
  # Make sure you also define a function for evaluating on the validation
  # set so that you can track performance over time

  # Create a local session to run the training.
  with tf.Session() as sess:
    # DONE
    # Make sure you initialize all variables before starting the tensorflow training
    sess.run(tf.initialize_all_variables())
    counter = 0
    plot_step = []
    train_error_list = []
    validation_error_list = []

    for step in xrange((num_epochs * train_size) // BATCH_SIZE):
        lowerLimit = ((counter) * BATCH_SIZE) % (train_size)
        upperLimit = ((counter+1) * BATCH_SIZE) % (train_size)
        if(upperLimit<lowerLimit):
          upperLimit = train_size
          lowerLimit = train_size - BATCH_SIZE
          counter = 0
        counter += 1
        batch_data = train_data[lowerLimit:upperLimit,]
        batch_label =  train_labels[lowerLimit:upperLimit]

        train_step.run(feed_dict={train_data_node: batch_data, train_labels_node: batch_label})
        # Loop through training steps here
        # HINT: always use small batches for training (as in SGD in your last exercise)
        # WARNING: The dataset does contain quite a few images if you want to test something quickly
        #          It might be useful to only train on a random subset!
        # For example use something like :
        # for step in max_steps:
        # Hint: make sure to evaluate your model every once in a while
        # For example like so:
        #print("print")
        if step % EVAL_FREQUENCY == 0:
          train_error = error.eval(feed_dict={train_data_node: batch_data, train_labels_node: batch_label})
          validation_error = eval_batches(validation_data, validation_labels)
          train_error_list.append(train_error)
          validation_error_list.append(validation_error)
          plot_step.append(int(step/100))
          print('Train error: ', train_error )
          print('Validation error: {}'.format(validation_error))

        # Finally, after the training! calculate the test result!
        # WARNING: You should never use the test result to optimize
        # your hyperparameters/network architecture, only look at the validation error to avoid
        # overfitting. Only calculate the test error in the very end for your best model!
        # if test_this_model_after_training:
    pickle.dump(train_error_list, open("train_D.p", "wb"))
    pickle.dump(plot_step, open("plot_D.p", "wb"))
    pickle.dump(validation_error_list, open("validation_D.p", "wb"))
    test_error = eval_batches(test_data, test_labels)
    print('Test error: {}'.format(test_error))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--use_rgbd',
      default=True,
      help='Use rgb-d input data (4 channels).',
      action='store_true'
  )
  parser.add_argument(
      '--self_test',
      default=False,
      action='store_true',
      help='True if running a self test.'
  )
  FLAGS = parser.parse_args()

  tf.app.run()
