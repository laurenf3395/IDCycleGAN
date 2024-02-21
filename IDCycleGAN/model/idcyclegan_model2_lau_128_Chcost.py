from __future__ import division, print_function

import time
import tensorflow as tf

from utils.layers import conv2d, conv2d_transpose, conv3d, conv3d_transpose, dis_block, dis_block2d,\
    linear, Conv2D, Deconv2D, Conv3D, Deconv3D, Batchnorm, Batchnorm3D
from utils.utils128 import sampleBatch, saveGIFBatch, write_image

import functools


class idcyclegan_model2(object):
    def __init__(self,
                 image_batch,
                 video_batch,
                 batch_size=32,
                 frame_size=32,
                 crop_size=64,
                 crop_size_img=128, #L:added
                 learning_rate=0.0002,
                 beta1=0.5,
                 critic_iterations=5):
        self.critic_iterations = critic_iterations
        self.crop_size = crop_size
        self.crop_size_img = crop_size_img #L:added
        self.beta1 = beta1
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.frame_size = frame_size
        self.images = image_batch
        self.videos = video_batch
        self.build_model()
        # self.REP = 2


    # def Normalize(name, axes, inputs):
    #     if axes != [0, 2, 3]:
    #         raise Exception('Layernorm over non-standard axes is unsupported')
    #     # return Layernorm(name, [1, 2, 3], inputs)
    #     return Batchnorm(name, [0, 2, 3], inputs, fused=True)
    #
    # def Normalize3D(name, axes, inputs):
    #     if axes != [0, 1, 2, 3]:
    #         raise Exception('Layernorm over non-standard axes is unsupported')
    #     return Batchnorm3D(name, [0, 1, 2, 3], inputs)

    # def nonlinearity(x):
    #     return tf.nn.relu(x)

    def upscale(*args, **kwargs):
        kwargs['output_dim'] = kwargs['output_dim']
        output = Conv3D(*args, **kwargs)
        output = tf.transpose(output, [0, 2, 3, 4, 1])
        # output is shape [batch, frame, height, width, channel]
        shape = tf.shape(output)
        output = tf.concat([tf.tile(
            tf.expand_dims(tf.image.resize_nearest_neighbor(output[:, i, :, :, :], [shape[2] * 2, shape[3] * 2]),
                           dim=1),
            multiples=[1, 2, 1, 1, 1]) for i in range(kwargs['upFrames'])], axis=1)
        output = tf.transpose(output, [0, 4, 1, 2, 3])
        return output

    def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
        output = Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
        output = tf.add_n(
            [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
        return output

    def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
        output = inputs
        output = tf.add_n(
            [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
        output = Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
        return output

    def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
        output = inputs
        output = tf.concat([output, output, output, output], axis=1)
        output = tf.transpose(output, [0, 2, 3, 1])
        output = tf.depth_to_space(output, 2)
        output = tf.transpose(output, [0, 3, 1, 2])
        output = Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
        return output

    def ResidualBlock(self, name, input_dim, output_dim, filter_size, inputs, resample=None, no_dropout=False, labels=None):
        """
        resample: None, 'down', or 'up'
        """
        if resample == 'down':
            conv_1 = functools.partial(Conv2D, input_dim=input_dim, output_dim=input_dim)
            conv_2 = functools.partial(self.ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
            conv_shortcut = self.ConvMeanPool
        elif resample == 'up':
            conv_1 = functools.partial(self.UpsampleConv, input_dim=input_dim, output_dim=output_dim)
            conv_shortcut = self.UpsampleConv
            conv_2 = functools.partial(Conv2D, input_dim=output_dim, output_dim=output_dim)
        elif resample == None:
            conv_shortcut = Conv2D
            conv_1 = functools.partial(Conv2D, input_dim=input_dim, output_dim=output_dim)
            conv_2 = functools.partial(Conv2D, input_dim=output_dim, output_dim=output_dim)
        else:
            raise Exception('invalid resample value')

        if output_dim == input_dim and resample == None:
            shortcut = inputs  # Identity skip-connection
        else:
            shortcut = conv_shortcut(name + '.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                     he_init=False, biases=True, inputs=inputs)

        output = inputs
        # output = self.Normalize(name + '.N1', output, labels=labels)
        output = Batchnorm(name+ '.N1', [0, 2, 3], output, fused=True)

        # output = self.nonlinearity(output)
        output = tf.nn.relu(output)

        output = conv_1(name + '.Conv1', filter_size=filter_size, inputs=output)
        # output = self.Normalize(name + '.N2', output, labels=labels)
        output = Batchnorm(name+ '.N2', [0, 2, 3], output, fused=True)

        # output = self.nonlinearity(output)
        output = tf.nn.relu(output)

        output = conv_2(name + '.Conv2', filter_size=filter_size, inputs=output)

        return shortcut + output

    def BottleneckResidualBlock3D(self, name, input_dim, output_dim, filter_size, inputs, upFrames=None, resample=None,
                                he_init=True):
        """
        resample: None, 'down', or 'up'
        """
        if resample == 'down':
            conv_shortcut = functools.partial(Conv3D, stride=2)
            conv_1 = functools.partial(Conv3D, input_dim=input_dim, output_dim=input_dim / 2)
            conv_1b = functools.partial(Conv3D, input_dim=input_dim / 2, output_dim=output_dim / 2, stride=2)
            conv_2 = functools.partial(Conv3D, input_dim=output_dim / 2, output_dim=output_dim)
        elif resample == 'up':
            conv_shortcut = self.upscale
            conv_1 = functools.partial(Conv3D, input_dim=input_dim, output_dim=input_dim / 2)
            conv_1b = functools.partial(Deconv3D, input_dim=input_dim / 2, output_dim=output_dim / 2)
            conv_2 = functools.partial(Conv3D, input_dim=output_dim / 2, output_dim=output_dim)
        elif resample == None:
            conv_shortcut = Conv3D
            conv_1 = functools.partial(Conv3D, input_dim=input_dim, output_dim=input_dim / 2)
            conv_1b = functools.partial(Conv3D, input_dim=input_dim / 2, output_dim=output_dim / 2)
            conv_2 = functools.partial(Conv3D, input_dim=input_dim / 2, output_dim=output_dim)

        else:
            raise Exception('invalid resample value')

        if output_dim == input_dim and resample == None:
            shortcut = inputs  # Identity skip-connection
        elif resample == 'up':
            shortcut = conv_shortcut(name + '.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                     he_init=False, biases=True, inputs=inputs, upFrames=upFrames)
        else:
            shortcut = conv_shortcut(name + '.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                     he_init=False, biases=True, inputs=inputs)

        output = inputs
        output = tf.nn.relu(output)
        output = conv_1(name + '.Conv1', filter_size=1, inputs=output, he_init=he_init)
        output = tf.nn.relu(output)
        output = conv_1b(name + '.Conv1B', filter_size=filter_size, inputs=output, he_init=he_init)
        output = tf.nn.relu(output)
        output = conv_2(name + '.Conv2', filter_size=1, inputs=output, he_init=he_init, biases=False)
        # output = self.Normalize3D(name + '.BN', [0, 1, 2, 3], output)
        output = Batchnorm3D(name + '.BN', [0, 1, 2, 3], output)


        return shortcut + (0.3 * output)

    def generatorImg(self, vid_batch, reuse=False):

        with tf.variable_scope('gen_i', reuse=reuse) as vs:
            """ -----------------------------------------------------------------------------------
                ENCODER 
            ----------------------------------------------------------------------------------- """
            self.en_h0 = conv3d(vid_batch, 3, 64, name="enc_conv1")
            self.en_h0 = tf.nn.relu(tf.contrib.layers.batch_norm(self.en_h0))
            add_activation_summary(self.en_h0)
            print(self.en_h0.get_shape().as_list())

            self.en_h1 = conv3d(self.en_h0, 64, 128, name="enc_conv2")
            self.en_h1 = tf.contrib.layers.batch_norm(self.en_h1, scope="enc_bn2")
            self.en_h1 = tf.nn.relu(self.en_h1)
            add_activation_summary(self.en_h1)
            print(self.en_h1.get_shape().as_list())

            # self.en_h2 = conv3d(self.en_h1, 128, 256, name="enc_conv3")
            # self.en_h2 = tf.contrib.layers.batch_norm(self.en_h2, scope="enc_bn3")
            # self.en_h2 = tf.nn.relu(self.en_h2)
            # add_activation_summary(self.en_h2)
            # print(self.en_h2.get_shape().as_list())

            """ -----------------------------------------------------------------------------------
            3D ResNet: 1. no "NCHW" format for 3DConv; 2. For resample = 'down', Current implementation 
            does not yet support strides in the batch and depth dimensions.
            ----------------------------------------------------------------------------------- """
            output = self.en_h1
            for i in xrange(3):
                output = self.BottleneckResidualBlock3D('res.16x16_{}'.format(i), 128, 128, 3, output, resample=None)
            # output = self.BottleneckResidualBlock3D('res.Down3', 128, 256, 3, output, resample='down')
            # print(output)

            self.en_h1 = output
            self.en_h2 = conv3d(self.en_h1, 128, 256, name="enc_conv3")
            self.en_h2 = tf.contrib.layers.batch_norm(self.en_h2, scope="enc_bn3")
            self.en_h2 = tf.nn.relu(self.en_h2)
            add_activation_summary(self.en_h2)
            print(self.en_h2.get_shape().as_list())

            output = self.en_h2
            for i in xrange(3):
                output = self.BottleneckResidualBlock3D('res.16x16_2_{}'.format(i), 256, 256, 3, output, resample=None)
                # output = self.BottleneckResidualBlock('res.Down4', 256, 512, 3, output, resample='down')
                # print(output)

            self.en_h2 = output
            self.en_h3 = conv3d(self.en_h2, 256, 512, name="enc_conv4")
            self.en_h3 = tf.contrib.layers.batch_norm(self.en_h3, scope="enc_bn4")
            self.en_h3 = tf.nn.relu(self.en_h3)
            add_activation_summary(self.en_h3)
            print(self.en_h3.get_shape().as_list())

            output = self.en_h3
            for i in xrange(3):
                output = self.BottleneckResidualBlock3D('res.16x16_3_{}'.format(i), 512, 512, 3, output, resample=None)

            '''self.en_h3 = output
            self.en_h4 = conv3d(self.en_h3, 512, 1024, name="enc_conv5")
            self.en_h4 = tf.contrib.layers.batch_norm(self.en_h4, scope="enc_bn5")
            self.en_h4 = tf.nn.relu(self.en_h4)
            add_activation_summary(self.en_h4)
            print(self.en_h4.get_shape().as_list())

            output = self.en_h4
            for i in xrange(3):
                output = self.BottleneckResidualBlock3D('res.16x16_4_{}'.format(i),1024, 1024, 3, output, resample=None)'''
            """ -----------------------------------------------------------------------------------
            Modification
            ----------------------------------------------------------------------------------- """
            # self.en_h2 = tf.transpose(output, [0, 2, 3, 4, 1])
            self.en_h3 = output

            # self.en_h3 = conv3d(self.en_h2, 256, 512, k_t=4, k_h=4, k_w=4, d_t=4, d_w=2, d_h=2, name="enc_conv4")
            self.en_h4 = conv3d(self.en_h3, 512, 1024, k_t=4, k_h=4, k_w=4, d_t=8, d_w=2, d_h=2, name="enc_conv5")

            self.en_h4 = tf.contrib.layers.batch_norm(self.en_h4, scope="enc_bn5")
            self.en_h4 = tf.nn.relu(self.en_h4)
            add_activation_summary(self.en_h4)
            print(self.en_h4.get_shape().as_list())

            """ -----------------------------------------------------------------------------------
            DECODER 
            ----------------------------------------------------------------------------------- """

            # self.fg_h0 = tf.reshape(self.en_h5, [-1, 4, 4, 2048])
            self.fg_h0 = tf.reshape(self.en_h4, [-1, 2, 2, 1024])
            print(self.fg_h0.get_shape().as_list())

            # self.fg_h1 = conv2d_transpose(self.fg_h0, 2048, [self.batch_size, 8, 8, 1024], name='g_f_h1')
            # self.fg_h1 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h1, scope='g_f_bn1'), name='g_f_relu1')
            # add_activation_summary(self.fg_h1)
            # print(self.fg_h1.get_shape().as_list())

            self.fg_h1 = conv2d_transpose(self.fg_h0, 1024, [self.batch_size, 4, 4, 512], name='g_f_h1')
            self.fg_h1 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h1, scope='g_f_bn1'), name='g_f_relu1')
            add_activation_summary(self.fg_h1)
            print(self.fg_h1.get_shape().as_list())

            self.fg_h2 = conv2d_transpose(self.fg_h1, 512, [self.batch_size, 8, 8, 256], name='g_f_h2')
            self.fg_h2 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h2, scope='g_f_bn2'), name='g_f_relu2')
            add_activation_summary(self.fg_h2)
            print(self.fg_h2.get_shape().as_list())

            self.fg_h3 = conv2d_transpose(self.fg_h2, 256, [self.batch_size, 16, 16, 128], name='g_f_h3')
            self.fg_h3 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h3, scope='g_f_bn3'), name='g_f_relu3')
            add_activation_summary(self.fg_h3)
            print(self.fg_h3.get_shape().as_list())

            self.fg_h4 = conv2d_transpose(self.fg_h3, 128, [self.batch_size, 32, 32, 64], name='g_f_h4')
            self.fg_h4 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h4, scope='g_f_bn4'), name='g_f_relu4')
            add_activation_summary(self.fg_h4)
            print(self.fg_h4.get_shape().as_list())

            self.fg_h5 = conv2d_transpose(self.fg_h4, 64, [self.batch_size, 64, 64, 32], name='g_f_h5')
            self.fg_h5 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h5, scope='g_f_bn5'), name='g_f_relu5')
            add_activation_summary(self.fg_h5)
            print(self.fg_h5.get_shape().as_list())

            self.fg_h6 = conv2d_transpose(self.fg_h5, 32, [self.batch_size, 128, 128, 3], name='g_f_h6')
            self.fg_img = tf.nn.tanh(self.fg_h6, name='g_f_actvcation')
            print(self.fg_img.get_shape().as_list())

        # gen_reg = tf.reduce_mean(tf.square(self.unmasked_video - self.fg_fg))

        # variables = tf.contrib.framework.get_variables(vs)
        # # return self.fg_fg, gen_reg, variables
        # return self.fg_img, variables
        return self.fg_img

    def generatorVid(self, img_batch, reuse=False):
        with tf.variable_scope('gen_v', reuse=reuse) as vs:
            """ -----------------------------------------------------------------------------------
            ENCODER 
            ----------------------------------------------------------------------------------- """

            self.en_h0 = conv2d(img_batch, 3, 64, k_h=4, k_w=4, d_w=2, d_h=2, name="enc_conv1")
            self.en_h0 = tf.nn.relu(tf.contrib.layers.batch_norm(self.en_h0))
            add_activation_summary(self.en_h0)
            print(self.en_h0.get_shape().as_list())

            self.en_h1 = conv2d(self.en_h0, 64, 128, k_h=4, k_w=4, d_w=2, d_h=2, name="enc_conv2")
            self.en_h1 = tf.contrib.layers.batch_norm(self.en_h1, scope="enc_bn2")
            self.en_h1 = tf.nn.relu(self.en_h1)
            add_activation_summary(self.en_h1)
            print(self.en_h1.get_shape().as_list())

            output = tf.transpose(self.en_h1, [0, 3, 1, 2])
            for i in xrange(3):
                output = self.ResidualBlock('res1.16x16_{}'.format(i), 128, 128, 3, output, resample=None)
            self.en_h1 = tf.transpose(output, [0, 2, 3, 1])

            self.en_h2 = conv2d(self.en_h1, 128, 256, k_h=4, k_w=4, d_w=2, d_h=2, name="enc_conv3")
            self.en_h2 = tf.contrib.layers.batch_norm(self.en_h2, scope="enc_bn3")
            self.en_h2 = tf.nn.relu(self.en_h2)
            add_activation_summary(self.en_h2)
            print(self.en_h2.get_shape().as_list())

            output = tf.transpose(self.en_h2, [0, 3, 1, 2])
            for i in xrange(3):
                output = self.ResidualBlock('res1.16x16_2_{}'.format(i), 256, 256, 3, output, resample=None)
            self.en_h2 = tf.transpose(output, [0, 2, 3, 1])

            self.en_h3 = conv2d(self.en_h2, 256, 512, k_h=4, k_w=4, d_w=2, d_h=2, name="enc_conv4")
            self.en_h3 = tf.contrib.layers.batch_norm(self.en_h3, scope="enc_bn4")
            self.en_h3 = tf.nn.relu(self.en_h3)
            add_activation_summary(self.en_h3)
            print(self.en_h3.get_shape().as_list())

            output = tf.transpose(self.en_h3, [0, 3, 1, 2])
            for i in xrange(3):
                output = self.ResidualBlock('res1.16x16_3_{}'.format(i), 512, 512, 3, output, resample=None)
            self.en_h3 = tf.transpose(output, [0, 2, 3, 1])

            self.en_h4 = conv2d(self.en_h3, 512, 1024, k_h=4, k_w=4, d_w=2, d_h=2, name="enc_conv5")
            self.en_h4 = tf.contrib.layers.batch_norm(self.en_h4, scope="enc_bn5")
            self.en_h4 = tf.nn.relu(self.en_h4)
            add_activation_summary(self.en_h4)
            print(self.en_h4.get_shape().as_list())

            """ -----------------------------------------------------------------------------------
            GENERATOR 
            ----------------------------------------------------------------------------------- """
            self.z_ = tf.reshape(self.en_h4, [self.batch_size, 2, 4, 4, 512])
            print(self.z_.get_shape().as_list())

            self.fg_h1 = conv3d_transpose(self.z_, 512, [self.batch_size, 4, 8, 8, 256], name='g_f_h1')
            self.fg_h1 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h1, scope='g_f_bn1'), name='g_f_relu1')
            add_activation_summary(self.fg_h1)
            print(self.fg_h1.get_shape().as_list())

            self.fg_h2 = conv3d_transpose(self.fg_h1, 256, [self.batch_size, 8, 16, 16, 128], name='g_f_h2')
            self.fg_h2 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h2, scope='g_f_bn2'), name='g_f_relu2')
            add_activation_summary(self.fg_h2)
            print(self.fg_h2.get_shape().as_list())

            self.fg_h3 = conv3d_transpose(self.fg_h2, 128, [self.batch_size, 16, 32, 32, 64], name='g_f_h3')
            self.fg_h3 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h3, scope='g_f_bn3'), name='g_f_relu3')
            add_activation_summary(self.fg_h3)
            print(self.fg_h3.get_shape().as_list())

            self.fg_h4 = conv3d_transpose(self.fg_h3, 64, [self.batch_size, 32, 64, 64, 3], name='g_f_h4')
            self.fg_vid = tf.nn.tanh(self.fg_h4, name='g_f_actvcation')
            print(self.fg_vid.get_shape().as_list())

            # gen_reg = tf.reduce_mean(tf.square(img_batch - self.fg_fg[:, 0, :, :, :]))

        # variables = tf.contrib.framework.get_variables(vs)
        # # return self.fg_fg, gen_reg, variables
        # return self.fg_vid, variables
        return self.fg_vid


    def discriminatorImg(self, image, reuse=False):
        with tf.variable_scope('disc_i', reuse=reuse) as vs:
            initial_dim = 128


            d_h0 = dis_block2d(image, 3, initial_dim, 'block1', reuse=reuse)
            d_h1 = dis_block2d(d_h0, initial_dim, initial_dim * 2, 'block2', reuse=reuse)
            d_h2 = dis_block2d(d_h1, initial_dim * 2, initial_dim * 4, 'block3', reuse=reuse)
            d_h3 = dis_block2d(d_h2, initial_dim * 4, initial_dim * 8, 'block4', reuse=reuse)
            d_h4 = dis_block2d(d_h3, initial_dim * 8, 1, 'block5', reuse=reuse, normalize=False)
            d_h5 = linear(tf.reshape(d_h4, [self.batch_size, -1]), 1)
        # variables = tf.contrib.framework.get_variables(vs)
        # return d_h5, variables
        return d_h5


    def discriminatorVid(self, video, reuse=False):
        with tf.variable_scope('disc_v', reuse=reuse) as vs:
            initial_dim = 64
            d_h0 = dis_block(video, 3, initial_dim, 'block1', reuse=reuse)
            d_h1 = dis_block(d_h0, initial_dim, initial_dim * 2, 'block2', reuse=reuse)
            d_h2 = dis_block(d_h1, initial_dim * 2, initial_dim * 4, 'block3', reuse=reuse)
            d_h3 = dis_block(d_h2, initial_dim * 4, initial_dim * 8, 'block4', reuse=reuse)
            d_h4 = dis_block(d_h3, initial_dim * 8, 1, 'block5', reuse=reuse, normalize=False)
            d_h5 = linear(tf.reshape(d_h4, [self.batch_size, -1]), 1)
        # variables = tf.contrib.framework.get_variables(vs)
        # return d_h5, variables
        return d_h5


    def computeCost(self, d_fake, d_real, images_fake, images, dim):
        g_cost = -tf.reduce_mean(d_fake)
        d_cost = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
        alpha = tf.random_uniform(
            shape=[self.batch_size, 1],
            minval=0.,
            maxval=1.
        )

        vid = tf.reshape(images, [self.batch_size, dim])
        fake = tf.reshape(images_fake, [self.batch_size, dim])
        differences = fake - vid
        interpolates = vid + (alpha * differences)
#L:changed to img
        if dim == self.crop_size_img * self.crop_size_img * 3:
            d_hat = self.discriminatorImg(tf.reshape(interpolates, [self.batch_size, self.crop_size_img,
                                                                self.crop_size_img, 3]), reuse=True)
        else:
            d_hat = self.discriminatorVid(tf.reshape(interpolates, [self.batch_size, self.frame_size, self.crop_size,
                                                                       self.crop_size, 3]), reuse=True)

        gradients = tf.gradients(d_hat, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        d_cost = d_cost + 10 * gradient_penalty

        return g_cost, d_cost

    def build_model(self):
        print("Setting up model...")

        self.videos_fake = self.generatorVid(self.images, reuse=False)
        self.images_fake_ = self.generatorImg(self.videos_fake, reuse=False)
        self.images_fake = self.generatorImg(self.videos, reuse=True)
        self.videos_fake_ = self.generatorVid(self.images_fake, reuse=True)

        self.d_real_vid = self.discriminatorVid(self.videos, reuse=False)
        self.d_real_img = self.discriminatorImg(self.images, reuse=False)
        self.d_fake_vid = self.discriminatorVid(self.videos_fake, reuse=True)
        self.d_fake_img = self.discriminatorImg(self.images_fake, reuse=True)

        self.reg_img = tf.reduce_mean(tf.square(self.images - self.images_fake_))
        self.reg_vid = tf.reduce_mean(tf.square(self.videos - self.videos_fake_))

        #L:Added lines
        self.img_reshape=tf.image.resize_images(self.images,[64,64])	
        self.img_fake_reshape=tf.image.resize_images(self.images_fake,[64,64])

        temp=[]
        temp2=[]
        for frame in range(32):
            self.reg_vid_new= tf.abs(self.img_fake_reshape - self.videos[:,frame,:,:,:])
            temp=tf.stack(self.reg_vid_new)
            self.reg_img_new= tf.abs(self.img_reshape - self.videos_fake[:,frame,:,:,:])
            temp2=tf.stack(self.reg_img_new)

        a=tf.reduce_min(temp)
        b=tf.reduce_min(temp2)
        self.reg_img_ = tf.reduce_mean(tf.square(b))
        self.reg_vid_ = tf.reduce_mean(tf.square(a))		
        #self.reg_img_ = tf.reduce_mean(tf.square(self.videos_fake[:, 0, :, :, :] - self.img_reshape))#L:changed self.images to #self.img_reshape
        #self.reg_vid_ = tf.reduce_mean(tf.square(self.img_fake_reshape - self.videos[:, 0, :, :, :]))

        #L: changed to crop_img
        dim_img = self.crop_size_img * self.crop_size_img * 3
        dim_vid = self.frame_size * self.crop_size * self.crop_size * 3
        self.g_cost_vid, self.d_cost_vid = self.computeCost(self.d_fake_vid, self.d_real_vid, self.videos_fake, self.videos, dim_vid)
        self.g_cost_img, self.d_cost_img = self.computeCost(self.d_fake_img, self.d_real_img, self.images_fake, self.images, dim_img)

        self.g_cost_final = self.g_cost_vid + self.g_cost_img + 1000*(self.reg_img + self.reg_vid) + 100*(self.reg_img_ + self.reg_vid_)

        self.d_cost_final = self.d_cost_vid + self.d_cost_img

        tf.summary.scalar("g_cost_final", self.g_cost_final)
        tf.summary.scalar("reg_img", self.reg_img)
        tf.summary.scalar("reg_vid", self.reg_vid)

        tf.summary.scalar("reg_img_", self.reg_img_)
        tf.summary.scalar("reg_vid_", self.reg_vid_)

        tf.summary.scalar("d_cost_final", self.d_cost_final)

        # Variables
        t_vars = tf.global_variables()
        g_vars = [var for var in t_vars if 'gen_' in var.name]
        d_vars = [var for var in t_vars if 'disc_' in var.name]


        self.d_adam, self.g_adam = None, None
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, beta2=0.999) \
                .minimize(self.d_cost_final, var_list=d_vars)
            self.g_adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, beta2=0.999) \
                .minimize(self.g_cost_final, var_list=g_vars)

        self.videos_sample = sampleBatch(self.videos, self.batch_size)
        self.videos_fake_sample = sampleBatch(self.videos_fake, self.batch_size)
        self.videos_fake_sample_ = sampleBatch(self.videos_fake_, self.batch_size)


        self.summary_op = tf.summary.merge_all()

    def _train(self, loss_val, var_list, optimizer):
        grads = optimizer.compute_gradients(loss_val, var_list=var_list)
        for grad, var in grads:
            add_gradient_summary(grad, var)
        return optimizer.apply_gradients(grads)

    def get_feed_dict(self, session):
        images = session.run(self.videos)[:, 0, :, :, :]
        feed_dict = {self.images: images}
        return feed_dict

    def train(self,
              session,
              step,
              step_i,
              summary_writer=None,
              log_summary=False,
              sample_dir=None,
              generate_sample=False):
        if log_summary:
            start_time = time.time()

        critic_itrs = self.critic_iterations

        for critic_itr in range(critic_itrs):
            session.run(self.d_adam)

        session.run(self.g_adam)

        if log_summary:
            g_cost_final, reg_img, reg_vid, reg_img_, reg_vid_,  d_cost_final, summary = session.run(
                [self.g_cost_final, self.reg_img, self.reg_vid, self.reg_img_, self.reg_vid_, self.d_cost_final, self.summary_op])
            summary_writer.add_summary(summary, step)
            print("Time: %g/itr, Step: %d, generator loss: %g (%g + %g + %g + %g), discriminator_loss: %g" % (
                time.time() - start_time, step, g_cost_final, reg_img, reg_vid, reg_img_, reg_vid_, d_cost_final))

        if generate_sample:
            images, videos_sample, images_gen, images_rec,  video_sample_gen,  video_sample_rec= \
                session.run([self.images, self.videos_sample, self.images_fake, self.images_fake_, self.videos_fake_sample, self.videos_fake_sample_])
            write_image(images, sample_dir, 'img_%d_gt.jpg' % step, rows=4)
            saveGIFBatch(videos_sample, sample_dir, 'vid_%d_gt' % step)
            write_image(images_gen, sample_dir, 'img_%d_gen.jpg' % step, rows=4)
            saveGIFBatch(video_sample_gen, sample_dir, 'vid_%d_gen' % step)
            write_image(images_rec, sample_dir, 'img_%d_rec.jpg' % step, rows=4)
            saveGIFBatch(video_sample_rec, sample_dir, 'vid_%d_rec' % step)


def add_activation_summary(var):
    tf.summary.histogram(var.op.name + "/activation", var)
    tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))


def add_gradient_summary(grad, var):
    if grad is not None:
        tf.summary.histogram(var.op.name + '/gradient', grad)
 
