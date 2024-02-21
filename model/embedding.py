from __future__ import division, print_function

import time
import tensorflow as tf

from utils.layers import conv2d, conv2d_transpose, conv3d, conv3d_transpose, dis_block, dis_block2d,\
    linear, Conv2D, Deconv2D, Conv3D, Deconv3D, Batchnorm, Batchnorm3D
from utils.utils128 import sampleBatch, saveGIFBatch, write_image

import functools

import models.inception_resnet_v1

    def computeFaceNetEmbedding(self, image_batch):
        # Build the inference graph
        # prelogits, _ = models.inception_resnet_v1.inference(image_batch, 0.8, phase_train=False, bottleneck_layer_size=128,
        #                                   weight_decay=0.0)
        prelogits, _ = models.inception_resnet_v1.inference(image_batch, 1, phase_train=False,
                                                            bottleneck_layer_size=128,
                                                            weight_decay=0.0,reuse=True)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        print(embeddings)
        return embeddings
