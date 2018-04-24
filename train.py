from utils import *
from Clock import get_random_clock
from Clock import get_random_clock_batch
from Clock import value_to_string
import tensorflow as tf
import os
from matplotlib import pyplot as plt
import Clock as Clock
import os


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    EPOCHS = 500000
    batch_size = 1
    learning_rate = 0.0001
    input_size = Clock.size

    color = False

    if color:
        input_channels = 3
    else:
        input_channels = 1

    logs_path = "./logs"

    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.25
    sess = tf.Session(config=config)

    x = tf.placeholder(tf.float32, [None, input_size, input_size, input_channels])
    y_ = tf.placeholder(tf.float32, [None, 1])
    y = deepnn_v1(x, input_size, input_channels)

    with tf.name_scope('Loss'):
        loss = tf.reduce_sum(tf.square(tf.subtract(y_, y)))
        train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    sess.run(init)

    checkpoint = tf.train.get_checkpoint_state("model")
    if checkpoint and checkpoint.model_checkpoint_path:
        s = saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Loaded model:", checkpoint.model_checkpoint_path)
        step = int(os.path.basename(checkpoint.model_checkpoint_path).split('-')[1])
    else:
        print("Can't find model")
        step = 0

    tf.summary.scalar("loss", loss)
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    cost_sum = 0
    acc_sum = 0

    print("Training starting at step: " + str(step) +
          ", Batch size: " + str(batch_size) +
          ", Learning rate: " + str(learning_rate))

    print_step = 10
    save_step = 1000
    try:
        for i in range(step, EPOCHS):

            img, value = get_random_clock(color)

            img = img.reshape(batch_size,input_size,input_size,input_channels)
            value = value.reshape(batch_size, 1)

            _, cost, prediction, summary = sess.run([train, loss, y, merged_summary_op],
                                           feed_dict={x: img, y_: value})

            summary_writer.add_summary(summary, i)
            cost_sum += cost

            if i % print_step == 0:
                if batch_size == 1:
                    print("Iteration : " + str(i) + ", Cost: " + str(cost_sum/print_step) +
                          ", value: " + value_to_string(value) +
                          ", prediction: " + value_to_string(prediction))
                else:
                    print("Iteration : " + str(i) + ", Cost: " + str(cost_sum / print_step) +
                          ", value: " + value_to_string(value[0]) +
                          ", prediction: " + value_to_string(prediction[0]))
                cost_sum = 0
                acc_sum = 0

            if i % save_step == 0:
                pass
                saver.save(sess, "./model/model.ckpt", global_step=i)

    except KeyboardInterrupt:
        print('Interrupted')


# tensorboard --logdir=/logs
