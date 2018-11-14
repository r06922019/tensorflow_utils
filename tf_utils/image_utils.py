#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


RGB_TO_XYZ = np.asarray( [  [0.412453, 0.357580, 0.180423],
                            [0.212671, 0.715160, 0.072169],
                            [0.019334, 0.119193, 0.950227]], dtype=np.float32 ).transpose()
RGB_TO_XYZ_TENSOR = tf.constant(RGB_TO_XYZ, dtype=tf.float32, name="rgb_to_xyz")


def tf_rgb_to_lab(rgb_images, srgb):
    with tf.name_scope("rgb_to_lab"):
        return tf_xyz_to_lab( tf_rgb_to_xyz(rgb_images, srgb) )


def tf_rgb_to_xyz(rgb_images, srgb=True):
    """
    rgb_images -> b, h, w, c tensor, float32, range(0, 1)
    
    # https://docs.opencv.org/ref/master/de/d25/imgproc_color_conversions.html

    ⎡ X ⎤   ⎡ 0.412453 0.357580 0.180423 ⎤   ⎡ R ⎤
    ⎢ Y ⎥ ← ⎢ 0.212671 0.715160 0.072169 ⎥ ⋅ ⎢ G ⎥
    ⎣ Z ⎦   ⎣ 0.019334 0.119193 0.950227 ⎦   ⎣ B ⎦
    
    """

    if srgb: # need to do inverse srgb transform
        with tf.name_scope("srgb_to_rgb"):
            SRGB_THRESHOLD = 0.04045
            srgb_a = 0.055
            # https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py#L683
            rgb_images = tf.where(rgb_images > SRGB_THRESHOLD, 
                tf.pow((rgb_images+srgb_a)/(1+srgb_a), 2.4) , rgb_images/12.92)

    with tf.name_scope("rgb_to_xyz"):
        
        shapes = tf.shape(rgb_images, name="input_shape")

        rgb_images = tf.clip_by_value(rgb_images, 0., 1., name="rgb_images_clip_0_1")
        rgb_images = tf.reshape(rgb_images, [-1, 3], name="reshape_for_matmul")
        
        xyz_images = tf.matmul(rgb_images, RGB_TO_XYZ_TENSOR, name="xyz_images")

        return tf.reshape(xyz_images, shapes, name="reshaped_back")


def tf_xyz_to_lab(xyz_images, 
    xyz_ref_white=[0.95047, 1., 1.08883],):
    
    # xyz_images : b, h, w, 3
    # "D65": {'2': (0.95047, 1., 1.08883),

    with tf.name_scope("xyz_to_lab"):

        with tf.name_scope("xyz_scaling"):
            xyz_images = xyz_images / xyz_ref_white
            xyz_images = tf.where( xyz_images > 0.008856, 
                tf.pow(xyz_images, 1. / 3.), 7.787 * xyz_images + 16. / 116. )
            x, y, z = xyz_images[..., 0], xyz_images[..., 1], xyz_images[..., 2]

        with tf.name_scope("L"):
            L = (116. * y) - 16.

        with tf.name_scope("a"):
            a = 500.0 * (x - y)
            
        with tf.name_scope("b"):
            b = 200.0 * (y - z)

        return tf.stack([L, a, b], axis=-1, name="lab_images")

