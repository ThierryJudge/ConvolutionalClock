import tensorflow as tf


def deepnn(x, input_size, input_channels):

  x_image = tf.reshape(x, [-1, input_size, input_size, input_channels])

  with tf.name_scope('conv_1'):
      W_conv1 = weight_variable([7, 7, input_channels, 32])
      b_conv1 = bias_variable([32])
      h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

      h_pool1 = max_pool_2x2(h_conv1)
      input_size = int(input_size / 2)

  with tf.name_scope('conv_2'):
      W_conv2 = weight_variable([5, 5, 32, 64])
      b_conv2 = bias_variable([64])
      h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

      h_pool2 = max_pool_2x2(h_conv2)
      input_size = int(input_size / 2)

  with tf.name_scope('conv_3'):
      W_conv3 = weight_variable([5, 5, 64, 128])
      b_conv3 = bias_variable([128])
      h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

      h_pool3 = max_pool_2x2(h_conv3)
      input_size = int(input_size / 2)

  with tf.name_scope('conv_4'):
      W_conv4 = weight_variable([3, 3, 128, 128])
      b_conv4 = bias_variable([128])
      h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)

      h_pool4 = max_pool_2x2(h_conv4)
      input_size = int(input_size / 2)

  with tf.name_scope('fc_1'):
      W_fc1 = weight_variable([input_size * input_size * 128, 1024])
      b_fc1 = bias_variable([1024])

      h_pool_flat = tf.reshape(h_pool4, [-1, input_size * input_size * 128])
      h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

  with tf.name_scope('fc_2'):
      W_fc2 = weight_variable([1024, 32])
      b_fc2 = bias_variable([32])

      h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

  # with tf.name_scope('fc_3'):
  #     W_fc3 = weight_variable([1024, 32])
  #     b_fc3 = bias_variable([32])
  #
  #     h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

  with tf.name_scope('fc_3'):
      W_fc4 = weight_variable([32, 1])
      b_fc4 = bias_variable([1])

      y_conv = tf.nn.relu(tf.matmul(h_fc2, W_fc4) + b_fc4)

  return y_conv


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.5)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.5, shape=shape)
  return tf.Variable(initial)


# input size = 200
def deepnn_v1(x, input_size, input_channels):

  x_image = tf.reshape(x, [-1, input_size, input_size, input_channels])

  with tf.name_scope('conv_1'):
      W_conv1 = weight_variable([7, 7, input_channels, 32])
      b_conv1 = bias_variable([32])
      h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

      h_pool1 = max_pool_2x2(h_conv1)
      input_size = int(input_size / 2)

  with tf.name_scope('conv_2'):
      W_conv2 = weight_variable([5, 5, 32, 64])
      b_conv2 = bias_variable([64])
      h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

      h_pool2 = max_pool_2x2(h_conv2)
      input_size = int(input_size / 2)

  with tf.name_scope('conv_3'):
      W_conv3 = weight_variable([5, 5, 64, 128])
      b_conv3 = bias_variable([128])
      h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

      h_pool3 = max_pool_2x2(h_conv3)
      input_size = int(input_size / 2)

  with tf.name_scope('fc_1'):
      W_fc1 = weight_variable([input_size * input_size * 128, 1024])
      b_fc1 = bias_variable([1024])

      h_pool3_flat = tf.reshape(h_pool3, [-1, input_size * input_size * 128])
      h_fc1 = tf.sigmoid(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

  with tf.name_scope('fc_2'):
      W_fc2 = weight_variable([1024, 32])
      b_fc2 = bias_variable([32])

      h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

  with tf.name_scope('fc_4'):
      W_fc4 = weight_variable([32, 1])
      b_fc4 = bias_variable([1])

      y_conv = tf.nn.relu(tf.matmul(h_fc2, W_fc4) + b_fc4)


  return y_conv
