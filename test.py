from utils import *
from Clock import get_random_clock
from Clock import value_to_string
import tensorflow as tf
import os
import cv2
import Clock
import math


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    VISUAL_TESTS = 10
    TESTS = 100
    color = False
    input_size = Clock.size

    if(color):
        input_channels = 3
    else:
        input_channels = 1

    tf.reset_default_graph()
    sess = tf.Session()

    x = tf.placeholder(tf.float32, [None, input_size, input_size, input_channels])
    y = deepnn_v1(x,input_size, input_channels)

    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    sess.run(init)

    checkpoint = tf.train.get_checkpoint_state("model_v1")
    if checkpoint and checkpoint.model_checkpoint_path:
        s = saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Loaded model:", checkpoint.model_checkpoint_path)
        step = int(os.path.basename(checkpoint.model_checkpoint_path).split('-')[1])
    else:
        print("Can't find model")

    total_error = 0

    for i in range(TESTS):
        img, value = get_random_clock(color)

        prediction = sess.run(y, feed_dict={x: img.reshape(1,input_size,input_size,input_channels)})

        error = abs(float(value) - float(prediction))
        total_error += error

    print("Average error for {} tests: {} minutes".format(TESTS, (total_error/TESTS)))

    for i in range(VISUAL_TESTS):
        img, value = get_random_clock(color)

        prediction = sess.run(y, feed_dict={x: img.reshape(1,input_size,input_size,input_channels)})

        print("Test " + str(i))
        print("Target: " + value_to_string(value))
        print("Prediction: " + value_to_string(prediction))

        cv2.imshow("Test {} \nTarget: {}, Prediction: {}".format(i,value_to_string(value), value_to_string(prediction)), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
