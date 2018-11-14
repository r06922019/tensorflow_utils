#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, sys
import tensorflow as tf
import numpy as np


def parse_arg():
    parser = argparse.ArgumentParser()
    targets = ["rgb_to_lab", "rgb_to_xyz"]
    parser.add_argument("--target", default=targets[0], help="Test targets", 
        choices=targets)
    args = parser.parse_args()
    print(args)
    return args


def test_rgb_to_lab():

    import skimage

    iterations = 10000
    batch_size, h, w, c = 16, 64, 64, 3

    from tf_utils.image_utils import tf_rgb_to_lab

    rgb_placeholder = tf.placeholder(tf.float32, shape=[batch_size, h, w, c], 
        name="rgb_placeholder")
    tf_xyz_tensor = tf_rgb_to_lab(rgb_placeholder, srgb=True)

    with tf.Session() as sess:
        for i in range(iterations):
            rgb_images = np.random.random((batch_size, h, w, c)).astype(np.float32)

            ski_xyz = np.asarray([ skimage.color.rgb2lab(rgb_images[_i]) for _i in range(batch_size) ])
            tf_xyz = sess.run(tf_xyz_tensor, feed_dict={rgb_placeholder:rgb_images})

            mse = np.mean(np.square(ski_xyz - tf_xyz))

            if mse < 1e-5: # results equal
                sys.stdout.write("\r>> Checked %d/%d " % (i + 1, iterations))
                sys.stdout.flush()
            else:
                print("\nError : %f \nToo bad." % (mse))
                print(ski_xyz[0,:3,:3,:])
                print(tf_xyz[0,:3,:3,:])
                quit()

    print("\nDone.")
    
    return


def test_rgb_to_xyz():

    import skimage

    iterations = 10000
    batch_size, h, w, c = 16, 64, 64, 3

    from tf_utils.image_utils import tf_rgb_to_xyz

    rgb_placeholder = tf.placeholder(tf.float32, shape=[batch_size, h, w, c], 
        name="rgb_placeholder")
    tf_xyz_tensor = tf_rgb_to_xyz(rgb_placeholder)

    with tf.Session() as sess:
        for i in range(iterations):
            rgb_images = np.random.random((batch_size, h, w, c)).astype(np.float32)

            ski_xyz = np.asarray([ skimage.color.rgb2xyz(rgb_images[_i]) for _i in range(batch_size) ])
            tf_xyz = sess.run(tf_xyz_tensor, feed_dict={rgb_placeholder:rgb_images})

            mse = np.mean(np.square(ski_xyz - tf_xyz))

            if mse < 1e-5: # results equal
                sys.stdout.write("\r>> Checked %d/%d " % (i + 1, iterations))
                sys.stdout.flush()
            else:
                print("\nError : %f \nToo bad." % (mse))
                quit()

    print("\nDone.")
    
    return

def show_on_tb(dir_name="show_graph_logs"):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        writer = tf.summary.FileWriter(dir_name, sess.graph)
        writer.close()
    print("python3 -m tensorboard.main --logdir=" + dir_name + " --port=32424") 
    return


if __name__ == "__main__":
    
    args = parse_arg()
    print("args.target =", args.target)
    
    target_func = "test_" + args.target
    if target_func in globals():
        globals()[ target_func ]()
    else:
        print("Function %s NOT FOUND" %(target_func))

    quit()


