#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


RGB_TO_XYZ = np.asarray( [  [0.412453, 0.357580, 0.180423],
                            [0.212671, 0.715160, 0.072169],
                            [0.019334, 0.119193, 0.950227]], dtype=np.float32 ).transpose()
RGB_TO_XYZ_TENSOR = tf.constant(RGB_TO_XYZ, dtype=tf.float32, name="rgb_to_xyz")


def tf_rgb_to_xyz(rgb_images, srgb=True):
    """
    rgb_images -> b, h, w, c tensor, float32, range(0, 1)
    
    # https://docs.opencv.org/ref/master/de/d25/imgproc_color_conversions.html

    ⎡ X ⎤   ⎡ 0.412453 0.357580 0.180423 ⎤   ⎡ R ⎤
    ⎢ Y ⎥ ← ⎢ 0.212671 0.715160 0.072169 ⎥ ⋅ ⎢ G ⎥
    ⎣ Z ⎦   ⎣ 0.019334 0.119193 0.950227 ⎦   ⎣ B ⎦
    
    """

    if srgb: # need to do inverse srgb transform
        SRGB_THRESHOLD = 0.04045
        srgb_a = 0.055
        # https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py#L683
        rgb_images = tf.where(rgb_images > SRGB_THRESHOLD, 
            tf.pow((rgb_images+srgb_a)/(1+srgb_a), 2.4) , rgb_images/12.92)

    with tf.name_scope("tf_rgb_to_xyz"):
        
        shapes = tf.shape(rgb_images, name="input_shape")

        rgb_images = tf.clip_by_value(rgb_images, 0., 1., name="rgb_images_clip_0_1")
        rgb_images = tf.reshape(rgb_images, [-1, 3], name="reshape_for_matmul")
        
        xyz_images = tf.matmul(rgb_images, RGB_TO_XYZ_TENSOR, name="xyz_images")
        xyz_images = tf.clip_by_value(xyz_images, 0., 1., name="xyz_images_clip_0_1")

        return tf.reshape(xyz_images, shapes, name="reshaped_back")


