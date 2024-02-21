from __future__ import division, print_function

import time
import tensorflow as tf

from utils.layers import conv2d, conv2d_transpose, conv3d, conv3d_transpose, dis_block, dis_block2d,\
    linear, Conv2D, Deconv2D, Conv3D, Deconv3D, Batchnorm, Batchnorm3D
from utils.utils128 import sampleBatch, saveGIFBatch, write_image

import functools


import facenet
import models.inception_resnet_v1


class idcyclegan_model3(object):
    def __init__(self,
                 image_batch,
                 video_batch,
                 batch_size=16,
                 frame_size=32,
                 crop_size=64,
                 crop_size_img=128, #L:added
                 learning_rate=0.0001,
                 beta1=0.5,
                 critic_iterations=5,
                 facenet_model='models/facenet'):
        self.critic_iterations = critic_iterations
        self.crop_size = crop_size
        self.crop_size_img= crop_size_img
        self.beta1 = beta1
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.frame_size = frame_size
        self.images = image_batch
        self.videos = video_batch
        self.facenet_model = facenet_model
        self.build_model()

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
            print("Encoder for generating image")
            self.en_h0 = conv3d(vid_batch, 3, 64, name="enc_conv1")
            self.en_h0 = tf.nn.relu(tf.contrib.layers.batch_norm(self.en_h0))
            add_activation_summary(self.en_h0)
            print(self.en_h0.get_shape().as_list())
            #self.en_h0 is [32, 16, 32, 32, 64]

            self.en_h1 = conv3d(self.en_h0, 64, 128, name="enc_conv2")
            self.en_h1 = tf.contrib.layers.batch_norm(self.en_h1, scope="enc_bn2")
            self.en_h1 = tf.nn.relu(self.en_h1)
            add_activation_summary(self.en_h1)
            print(self.en_h1.get_shape().as_list())
            #self.en_h1 is [32, 8, 16, 16, 128]

            # self.en_h2 = conv3d(self.en_h1, 128, 256, name="enc_conv3")
            # self.en_h2 = tf.contrib.layers.batch_norm(self.en_h2, scope="enc_bn3")
            # self.en_h2 = tf.nn.relu(self.en_h2)
            # add_activation_summary(self.en_h2)
            # print(self.en_h2.get_shape().as_list())

            """ -----------------------------------------------------------------------------------
            3D ResNet: 1. no "NCHW" format for 3DConv; 2. For resample = 'down', Current implementation 
            does not yet support strides in the batch and depth dimensions.
            ----------------------------------------------------------------------------------- """


            #output = self.en_h1
            #for i in xrange(3):
            #    output = self.BottleneckResidualBlock3D('res.16x16_{}'.format(i), 128, 128, 3, output, resample=None)
            # output = self.BottleneckResidualBlock3D('res.Down3', 128, 256, 3, output, resample='down')
            # print(output)

            #self.en_h1 = output
            self.en_h2 = conv3d(self.en_h1, 128, 256, name="enc_conv3")
            self.en_h2 = tf.contrib.layers.batch_norm(self.en_h2, scope="enc_bn3")
            self.en_h2 = tf.nn.relu(self.en_h2)
            add_activation_summary(self.en_h2)
            print(self.en_h2.get_shape().as_list())
            #self.en_h2 is [32, 4, 8, 8, 256]

            #output = self.en_h2
            #for i in xrange(3):
            #    output = self.BottleneckResidualBlock3D('res.16x16_2_{}'.format(i), 256, 256, 3, output, resample=None)
            # output = self.BottleneckResidualBlock('res.Down4', 256, 512, 3, output, resample='down')
            # print(output)


            """ -----------------------------------------------------------------------------------
            Modification
            ----------------------------------------------------------------------------------- """
            # self.en_h2 = tf.transpose(output, [0, 2, 3, 4, 1])
            #self.en_h2 = output

            # self.en_h3 = conv3d(self.en_h2, 256, 512, k_t=4, k_h=4, k_w=4, d_t=4, d_w=2, d_h=2, name="enc_conv4")
            self.en_h3 = conv3d(self.en_h2, 256, 256, k_t=2, k_h=4, k_w=4, d_t=2, d_w=2, d_h=2, name="enc_conv4")
            self.en_h3 = tf.contrib.layers.batch_norm(self.en_h3, scope="enc_bn4")
            self.en_h3 = tf.nn.relu(self.en_h3)
            add_activation_summary(self.en_h3)
            print(self.en_h3.get_shape().as_list())
            #self.en_h3 is [32, 2, 4, 4, 256]

            self.en_h4 = conv3d(self.en_h3,256,256,name="enc_conv5")
            self.en_h4 = tf.contrib.layers.batch_norm(self.en_h4, scope="enc_bn5")
            self.en_h4 = tf.nn.relu(self.en_h4)
            add_activation_summary(self.en_h4)
            print(self.en_h4.get_shape().as_list())
            #self.en_h4 is [32, 1, 2, 2, 256]

            self.en_h5 = conv3d(self.en_h4, 256, 256, k_t=1, k_h=4, k_w=4, d_t=1, d_w=2, d_h=2,name="enc_conv6")
            self.en_h5 = tf.contrib.layers.batch_norm(self.en_h5, scope="enc_bn6")
            self.en_h5 = tf.nn.relu(self.en_h5)
            add_activation_summary(self.en_h5)
            print(self.en_h5.get_shape().as_list())
            #self.en_h5 is [32, 1, 1, 1, 256]


            """ -----------------------------------------------------------------------------------
            DECODER 
            ----------------------------------------------------------------------------------- """
            print("Decoder for generating image")
            self.fg_h0 = tf.reshape(self.en_h5, [-1, 1, 1, 256])
            print(self.fg_h0.get_shape().as_list())
            #self.fg_h0 is [32, 1, 1, 256]

            self.fg_h1 = conv2d_transpose(self.fg_h0, 256, [self.batch_size, 2, 2, 256], name='g_f_h1')
            self.fg_h1 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h1, scope='g_f_bn1'), name='g_f_relu1')
            add_activation_summary(self.fg_h1)
            print(self.fg_h1.get_shape().as_list())
            #self.fg_h1 is [32, 2, 2, 256]

            enc5= tf.reshape(self.en_h4,[-1,2,2,256])
            #enc5 is [32,2,2,256] and self.fg_h1 is [32,2,2,256] concatenating [32,2,2,512]
            enco5=tf.concat([self.fg_h1,enc5],axis=3)
            self.fg_h2= conv2d_transpose(enco5, 512, [self.batch_size, 4, 4, 256], name='g_f_h2')
            self.fg_h2 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h2, scope='g_f_bn2'), name='g_f_relu2')
            add_activation_summary(self.fg_h2)
            print(self.fg_h2.get_shape().as_list())
            #self.fg_h2 is now [32, 4, 4, 256]
	   
            enc4= tf.strided_slice(self.en_h3, [0,0,0,0,0],[16,2,4,4,256],[1,2,1,1,1])
            enco4=tf.reshape(enc4,[-1,4,4,256])
            #enco4 is [32,4,4,256] and self.fg_h2 is [32,4,4,256] concatenating [32,4,4,512]
            encod4=tf.concat([self.fg_h2,enco4],axis=3)
            self.fg_h3 = conv2d_transpose(encod4, 512, [self.batch_size, 8, 8, 256], name='g_f_h3')
            self.fg_h3 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h3, scope='g_f_bn3'), name='g_f_relu3')
            add_activation_summary(self.fg_h3)
            print(self.fg_h3.get_shape().as_list())
            #self.fg_h3 is now [32, 8, 8, 256]

            enc3= tf.strided_slice(self.en_h2, [0,0,0,0,0],[16,4,8,8,256],[1,4,1,1,1])
            enco3=tf.reshape(enc3,[-1,8,8,256])
            #enco3 is [32,8,8,256] and self.fg_h3 is [32,8,8,256] concatenating [32,8,8,512]
            encod3=tf.concat([self.fg_h3,enco3],axis=3)
            self.fg_h4 = conv2d_transpose(encod3, 512, [self.batch_size, 16, 16, 128], name='g_f_h4')
            self.fg_h4 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h4, scope='g_f_bn4'), name='g_f_relu4')
            add_activation_summary(self.fg_h4)
            print(self.fg_h4.get_shape().as_list())
            #self.fg_h4 is now [32, 16, 16, 128]

            enc2= tf.strided_slice(self.en_h1, [0,0,0,0,0],[16,8,16,16,128],[1,8,1,1,1])
            enco2=tf.reshape(enc2,[-1,16,16,128])
            #enco2 is [32,16,16,128] and self.fg_h4 is [32,16,16,128] concatenating [32,16, 16,256]
            encod2=tf.concat([self.fg_h4,enco2],axis=3)
            self.fg_h5 = conv2d_transpose(encod2, 256, [self.batch_size, 32, 32, 64], name='g_f_h5')
            self.fg_h5 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h5, scope='g_f_bn5'), name='g_f_relu5')
            add_activation_summary(self.fg_h5)
            print(self.fg_h5.get_shape().as_list())
            #self.fg_h5 is now [32, 32, 32, 64]

            enc1= tf.strided_slice(self.en_h0, [0,0,0,0,0],[16,16,32,32,64],[1,16,1,1,1])
            enco1=tf.reshape(enc1,[-1,32,32,64])
            #enco1 is [32,32,32,64] and self.fg_h5 is [32,32,32,64] concatenating [32,32, 32,128]
            encod1=tf.concat([self.fg_h5,enco1],axis=3)
            self.fg_h6= conv2d_transpose(encod1, 128,[self.batch_size, 64, 64, 32], name='g_f_h6')
            self.fg_h6 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h6, scope='g_f_bn6'), name='g_f_relu6')
            add_activation_summary(self.fg_h6)
            print(self.fg_h6.get_shape().as_list())
            #self.fg_h6 is [32, 64, 64, 32]

            self.fg_h7= conv2d_transpose(self.fg_h6, 32,[self.batch_size, 128, 128, 3], name='g_f_h7')
            self.fg_img = tf.nn.tanh(self.fg_h7, name='g_f_actvcation')
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

            # self.en_h0 = conv2d(img_batch, 3, 64, k_h=4, k_w=4, d_w=2, d_h=2, name="enc_conv1")
            # self.en_h0 = tf.nn.relu(tf.contrib.layers.batch_norm(self.en_h0))
            # add_activation_summary(self.en_h0)
            # print(self.en_h0.get_shape().as_list())

            self.en_h0 = conv2d(img_batch, 3, 32, k_h=4, k_w=4, d_w=2, d_h=2, name="enc_conv1")
            self.en_h0 = tf.nn.relu(tf.contrib.layers.batch_norm(self.en_h0))
            add_activation_summary(self.en_h0)
            print(self.en_h0.get_shape().as_list())


            self.en_h1 = conv2d(self.en_h0, 32, 64, k_h=4, k_w=4, d_w=2, d_h=2, name="enc_conv2")
            self.en_h1 = tf.contrib.layers.batch_norm(self.en_h1, scope="enc_bn2")
            self.en_h1 = tf.nn.relu(self.en_h1)
            add_activation_summary(self.en_h1)
            print(self.en_h1.get_shape().as_list())


            #output = tf.transpose(self.en_h1, [0, 3, 1, 2])
            #for i in xrange(3):
            #    output = self.ResidualBlock('res1.16x16_{}'.format(i), 256, 256, 3, output, resample=None)
            #self.en_h1 = tf.transpose(output, [0, 2, 3, 1])

            self.en_h2 = conv2d(self.en_h1, 64, 128, k_h=4, k_w=4, d_w=2, d_h=2, name="enc_conv3")
            self.en_h2 = tf.contrib.layers.batch_norm(self.en_h2, scope="enc_bn3")
            self.en_h2 = tf.nn.relu(self.en_h2)
            add_activation_summary(self.en_h2)
            print(self.en_h2.get_shape().as_list())


            #output = tf.transpose(self.en_h2, [0, 3, 1, 2])
            #for i in xrange(3):
            #    output = self.ResidualBlock('res1.16x16_2_{}'.format(i), 512, 512, 3, output, resample=None)
            #self.en_h2 = tf.transpose(output, [0, 2, 3, 1])

            self.en_h3 = conv2d(self.en_h2, 128, 256, k_h=4, k_w=4, d_w=2, d_h=2, name="enc_conv4")
            self.en_h3 = tf.contrib.layers.batch_norm(self.en_h3, scope="enc_bn4")
            self.en_h3 = tf.nn.relu(self.en_h3)
            add_activation_summary(self.en_h3)
            print(self.en_h3.get_shape().as_list())

            self.en_h4 = conv2d(self.en_h3, 256, 256, k_h=4, k_w=4, d_w=2, d_h=2, name="enc_conv5")
            self.en_h4 = tf.contrib.layers.batch_norm(self.en_h4, scope="enc_bn5")
            self.en_h4 = tf.nn.relu(self.en_h4)
            add_activation_summary(self.en_h4)
            print(self.en_h4.get_shape().as_list())
            #self.en_h4 is [32, 4, 4, 256]

            self.en_h5 = conv2d(self.en_h4, 256, 256, k_h=4, k_w=4, d_w=2, d_h=2, name="enc_conv6")
            self.en_h5 = tf.contrib.layers.batch_norm(self.en_h5, scope="enc_bn6")
            self.en_h5 = tf.nn.relu(self.en_h5)
            add_activation_summary(self.en_h5)
            print(self.en_h5.get_shape().as_list())
            #self.en_h5 is [32,2,2,256]

            self.en_h6 = conv2d(self.en_h5, 256, 256, k_h=4, k_w=4, d_w=2, d_h=2, name="enc_conv7")
            self.en_h6 = tf.contrib.layers.batch_norm(self.en_h6, scope="enc_bn7")
            self.en_h6 = tf.nn.relu(self.en_h6)
            add_activation_summary(self.en_h6)
            print(self.en_h6.get_shape().as_list())
            #self.en_h6 is [32,1,1,256]

            """ -----------------------------------------------------------------------------------
            GENERATOR 
            ----------------------------------------------------------------------------------- """
            self.z_ = tf.reshape(self.en_h6, [self.batch_size, 1, 1, 1, 256])
            print(self.z_.get_shape().as_list())

            self.fg_h1 = conv3d_transpose(self.z_, 256, [self.batch_size, 1, 2, 2, 256], name='g_f_h1')
            self.fg_h1 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h1, scope='g_f_bn1'), name='g_f_relu1')
            add_activation_summary(self.fg_h1)
            print(self.fg_h1.get_shape().as_list())
            #self.fg_h1 is [32, 1, 2, 2, 256]

            encv5= tf.reshape(self.en_h5,[self.batch_size, 1,2,2,256])
            encov5 = tf.concat([self.fg_h1,encv5], axis=4)
            self.fg_h2 = conv3d_transpose(encov5,512, [self.batch_size, 2, 4, 4, 256], name='g_f_h2')
            self.fg_h2 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h2, scope='g_f_bn2'), name='g_f_relu2')
            add_activation_summary(self.fg_h2)			
            print(self.fg_h2.get_shape().as_list())


            encv4= tf.tile(tf.expand_dims(self.en_h4, axis=1),[1,2,1,1,1])
            encvo4= tf.reshape(encv4,[self.batch_size, 2,4,4,256])
            encodv4 = tf.concat([self.fg_h2,encvo4], axis=4)
            self.fg_h3 = conv3d_transpose(encodv4, 512, [self.batch_size, 4, 8, 8, 256], name='g_f_h3')
            self.fg_h3 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h3, scope='g_f_bn3'), name='g_f_relu3')
            add_activation_summary(self.fg_h3)
            print(self.fg_h3.get_shape().as_list())

            encv3= tf.tile(tf.expand_dims(self.en_h3, axis=1),[1,4,1,1,1])
            encvo3= tf.reshape(encv3,[self.batch_size, 4,8,8,256])
            encodv3 = tf.concat([self.fg_h3,encvo3], axis=4)
            self.fg_h4 = conv3d_transpose(encodv3, 512, [self.batch_size, 8, 16, 16, 128], name='g_f_h4')
            self.fg_h4 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h4, scope='g_f_bn4'), name='g_f_relu4')
            add_activation_summary(self.fg_h4)
            print(self.fg_h4.get_shape().as_list())

            encv2= tf.tile(tf.expand_dims(self.en_h2, axis=1),[1,8,1,1,1])
            encvo2= tf.reshape(encv2,[self.batch_size, 8,16,16,128])
            #encvo2 is [32,8,16,16,128] and self.fg_h4 is [32,8,16,16,128] ans concat is [32,8,16,16,256]
            encodv2 = tf.concat([self.fg_h4,encvo2], axis=4)
            self.fg_h5 = conv3d_transpose(encodv2, 256, [self.batch_size, 16, 32, 32, 64], name='g_f_h5')
            self.fg_h5 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h5, scope='g_f_bn5'), name='g_f_relu5')
            add_activation_summary(self.fg_h5)
            print(self.fg_h5.get_shape().as_list())
            #self.fg_h5 is [32, 16, 32, 32, 64]
			
            encv1= tf.tile(tf.expand_dims(self.en_h1, axis=1),[1,16,1,1,1])
            encvo1= tf.reshape(encv1,[self.batch_size, 16,32,32,64])
            encodv1 = tf.concat([self.fg_h5,encvo1], axis=4)
            self.fg_h6 = conv3d_transpose(encodv1, 128, [self.batch_size, 32, 64, 64,3], name='g_f_h6')
            self.fg_vid = tf.nn.tanh(self.fg_h6, name='g_f_actvcation')
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



    def computeFaceNetEmbedding(self, image_batch):
        # Build the inference graph
        # prelogits, _ = models.inception_resnet_v1.inference(image_batch, 0.8, phase_train=False, bottleneck_layer_size=128,
        #                                   weight_decay=0.0)
        prelogits, _ = models.inception_resnet_v1.inference(image_batch, 1, phase_train=False,
                                                            bottleneck_layer_size=128,
                                                            weight_decay=0.0,reuse=True)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        return embeddings

    def reduce_var(self, x, axis=None, keepdims=False):
        """Variance of a tensor, alongside the specified axis.
        # Arguments
            x: A tensor or variable.
            axis: An integer, the axis to compute the variance.
            keepdims: A boolean, whether to keep the dimensions or not.
                If `keepdims` is `False`, the rank of the tensor is reduced
                by 1. If `keepdims` is `True`,
                the reduced dimension is retained with length 1.
        # Returns
            A tensor with the variance of elements of `x`.
        """
        m = tf.reduce_mean(x, axis=axis, keep_dims=True)
        devs_squared = tf.square(x - m)
        return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)

    def reduce_std(self, x, axis=None, keep_dims=False):
        """Standard deviation of a tensor, alongside the specified axis.
        # Arguments
            x: A tensor or variable.
            axis: An integer, the axis to compute the standard deviation.
            keepdims: A boolean, whether to keep the dimensions or not.
                If `keepdims` is `False`, the rank of the tensor is reduced
                by 1. If `keepdims` is `True`,
                the reduced dimension is retained with length 1.
        # Returns
            A tensor with the standard deviation of elements of `x`.
        """
        return tf.sqrt(self.reduce_var(x, axis=axis, keepdims=keep_dims))


    def preWhiten(self, imgBatch, reshape_size):

        # mean = np.mean(x)
        # std = np.std(x)
        # std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        # y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        # return y

        imgBatch = tf.reshape(imgBatch, [self.batch_size, reshape_size*reshape_size*3]) #L:32 to 16

        mean = tf.reduce_mean(imgBatch, axis=1, keep_dims=True)
        std = self.reduce_std(imgBatch, axis=1, keep_dims=True)

        tmp = tf.reshape(1.0 / tf.sqrt(tf.cast(reshape_size*reshape_size*3, tf.float32)),[1,1])
        std_adj = tf.maximum(std, tf.tile(tmp, [self.batch_size, 1])) #L:32 to 16

        std_adj = tf.tile(std_adj, [1, reshape_size*reshape_size*3])
        mean = tf.tile(mean, [1, reshape_size*reshape_size*3])

        imgBatch = tf.div(tf.subtract(imgBatch, mean), std_adj)

        return tf.reshape(imgBatch, [self.batch_size, reshape_size, reshape_size, 3]) #L:32 to 16


    def preWhiten_vid(self, imgBatch, reshape_size):

        # mean = np.mean(x)
        # std = np.std(x)
        # std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        # y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        # return y

        imgBatch = tf.reshape(imgBatch, [self.batch_size*8, reshape_size*reshape_size*3]) #L:32 to 16

        mean = tf.reduce_mean(imgBatch, axis=1, keep_dims=True)
        std = self.reduce_std(imgBatch, axis=1, keep_dims=True)

        tmp = tf.reshape(1.0 / tf.sqrt(tf.cast(reshape_size*reshape_size*3, tf.float32)),[1,1])
        std_adj = tf.maximum(std, tf.tile(tmp, [self.batch_size*8, 1])) #L:32 to 16

        std_adj = tf.tile(std_adj, [1, reshape_size*reshape_size*3])
        mean = tf.tile(mean, [1, reshape_size*reshape_size*3])

        imgBatch = tf.div(tf.subtract(imgBatch, mean), std_adj)

        return tf.reshape(imgBatch, [self.batch_size*8, reshape_size, reshape_size, 3])#L:32 to 16


    def build_model(self):
        print("Setting up model...")

        # # self.facenet_ = importlib.import_module(models.inception_resnet_v1)
        # # Load the model
        facenet.load_model(self.facenet_model)
        self.reshape_size = 160


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

        #L:32 to 16
        self.videos_rgb = tf.clip_by_value(((tf.strided_slice(self.videos, [0,0,0,0,0], [self.batch_size,32,64,64,3], [1,4,1,1,1]) + 1.0) * 127.5), 0, 255)

        self.images_rgb = tf.clip_by_value(((self.images + 1.0) * 127.5), 0, 255)

        self.videos_fake_rgb = tf.clip_by_value(((tf.strided_slice(self.videos_fake, [0,0,0,0,0], [self.batch_size,32,64,64,3], [1,4,1,1,1]) + 1.0) * 127.5), 0, 255)
        self.images_fake_rgb = tf.clip_by_value(((self.images_fake + 1.0) * 127.5), 0, 255)
        #L:32 to 16
        self.videos_resize = tf.image.resize_images(tf.reshape(self.videos_rgb, [self.batch_size*8, 64, 64, 3]), [self.reshape_size, self.reshape_size])
        self.images_resize = tf.image.resize_images(self.images_rgb, [self.reshape_size, self.reshape_size])
        self.videos_fake_resize = tf.image.resize_images(tf.reshape(self.videos_fake_rgb, [self.batch_size*8, 64, 64, 3]), [self.reshape_size, self.reshape_size])
        self.images_fake_resize = tf.image.resize_images(self.images_fake_rgb, [self.reshape_size, self.reshape_size])

        self.videos_whiten = self.preWhiten_vid(self.videos_resize, self.reshape_size)
        self.images_whiten = self.preWhiten(self.images_resize, self.reshape_size)
        self.videos_fake_whiten = self.preWhiten_vid(self.videos_fake_resize, self.reshape_size)
        self.images_fake_whiten = self.preWhiten(self.images_fake_resize, self.reshape_size)

        self.emb_real_vid = tf.reshape(self.computeFaceNetEmbedding(self.videos_whiten), [self.batch_size, 8, 128])
        self.emb_real_img = self.computeFaceNetEmbedding(self.images_whiten)
        self.emb_fake_vid = tf.reshape(self.computeFaceNetEmbedding(self.videos_fake_whiten), [self.batch_size, 8, 128])
        self.emb_fake_img = self.computeFaceNetEmbedding(self.images_fake_whiten)
        temp=[]

        for frame in range(32):
            #self.reg_vid_new= self.images_fake
            self.reg_vid_new= tf.tile(tf.expand_dims(self.emb_fake_img, axis=1), [1, 8, 1]) - \
                                                               self.emb_real_vid
            temp=tf.stack(self.reg_vid_new)

        a=tf.reduce_min(temp)

        self.reg_img_ = tf.reduce_sum(tf.reduce_mean(tf.square(self.emb_fake_vid - \
                                                               tf.tile(tf.expand_dims(self.emb_real_img, axis=1), [1, 8, 1])), axis=1, keep_dims=True))
        self.reg_vid_ = tf.reduce_sum(tf.reduce_mean(tf.square(self.emb_real_vid- \
		                                             tf.tile(tf.expand_dims(self.emb_fake_img, axis=1), [1, 8, 1])), axis=1,keep_dims=True))
#L:crop_size_img
        dim_img = self.crop_size_img * self.crop_size_img * 3
        dim_vid = self.frame_size * self.crop_size * self.crop_size * 3
        self.g_cost_vid, self.d_cost_vid = self.computeCost(self.d_fake_vid, self.d_real_vid, self.videos_fake,
                                                            self.videos, dim_vid)
        self.g_cost_img, self.d_cost_img = self.computeCost(self.d_fake_img, self.d_real_img, self.images_fake,
                                                            self.images, dim_img)

        self.g_cost_final = self.g_cost_vid + self.g_cost_img + 1000*(self.reg_img + self.reg_vid) +1000*(self.reg_img_ + self.reg_vid_)

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
        self.facenet_vars = [var for var in t_vars if 'InceptionResnetV1' in var.name]


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

        if step_i == 0:
            print("Loading FaceNet...")

            for var in self.facenet_vars:
                print(var.name)
                var_intial = tf.get_default_graph().get_tensor_by_name(var.name)
                session.run(var.assign(var_intial))

        critic_itrs = self.critic_iterations

        for critic_itr in range(critic_itrs):
             session.run(self.d_adam)

        session.run(self.g_adam)

        if log_summary:
            g_cost_final, reg_img, reg_vid, reg_img_, reg_vid_, d_cost_final, summary = session.run(
                [self.g_cost_final, self.reg_img, self.reg_vid, self.reg_img_, self.reg_vid_, self.d_cost_final,
                 self.summary_op])
            summary_writer.add_summary(summary, step)
            print("Time: %g/itr, Step: %d, generator loss: %g (%g + %g + %g + %g), discriminator_loss: %g" % (
                time.time() - start_time, step, g_cost_final, reg_img, reg_vid, reg_img_, reg_vid_, d_cost_final))

        if generate_sample:
            images, videos_sample, images_gen, images_rec, video_sample_gen, video_sample_rec = \
                session.run(
                    [self.images, self.videos_sample, self.images_fake, self.images_fake_, self.videos_fake_sample,
                     self.videos_fake_sample_])
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
