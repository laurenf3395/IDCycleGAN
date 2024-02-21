import tensorflow as tf
import numpy as np
import tflib as lib

_default_weightnorm = False
def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True

_weights_stdev = None
def set_weights_stdev(weights_stdev):
    global _weights_stdev
    _weights_stdev = weights_stdev

def unset_weights_stdev():
    global _weights_stdev
    _weights_stdev = None
    _weights_stdev = None

def conv2d(input_, input_dim, output_dim,
           k_h=4, k_w=4, d_h=4, d_w=4, name="conv2d", padding="SAME"):  #L:changed from 2 to 4 for dh and dw
    def uniform(std_dev, size):
        return np.random.uniform(
            low=-std_dev * np.sqrt(3),
            high=std_dev * np.sqrt(3),
            size=size
        ).astype('float32')

    """ 
    init weights like in    
    "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
    """
    with tf.variable_scope(name):
        fan_in = input_dim * k_h * k_w
        fan_out = (output_dim * k_h * k_w) / (d_h * d_w)

        filters_std = np.sqrt(4. / (fan_in + fan_out))

        filter_values = uniform(
            filters_std,
            (k_h, k_w, input_dim, output_dim)
        )

        w_init = tf.Variable(filter_values, name='filters_init')
        w = tf.get_variable('filters', initializer=w_init.initialized_value())
        b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        # result = tf.nn.conv2d(
        #     input=input_,
        #     filter=w,
        #     strides=[1, d_h, d_w, 1],
        #     padding=padding,
        #     data_format='NHWC'
        # )
        result = tf.nn.conv2d(
            input=input_,
            filter=w,
            strides=[1, d_h, d_w, 1],
            padding=padding
        )
        result = tf.nn.bias_add(result, b)

    return result

def Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=True, mask_type=None, stride=1, weightnorm=None, biases=True, gain=1.):
    """
    inputs: tensor of shape (batch size, num channels, height, width)
    mask_type: one of None, 'a', 'b'

    returns: tensor of shape (batch size, num channels, height, width)
    """
    with tf.name_scope(name) as scope:

        if mask_type is not None:
            mask_type, mask_n_channels = mask_type

            mask = np.ones(
                (filter_size, filter_size, input_dim, output_dim),
                dtype='float32'
            )
            center = filter_size // 2

            # Mask out future locations
            # filter shape is (height, width, input channels, output channels)
            mask[center+1:, :, :, :] = 0.
            mask[center, center+1:, :, :] = 0.

            # Mask out future channels
            for i in xrange(mask_n_channels):
                for j in xrange(mask_n_channels):
                    if (mask_type=='a' and i >= j) or (mask_type=='b' and i > j):
                        mask[
                            center,
                            center,
                            i::mask_n_channels,
                            j::mask_n_channels
                        ] = 0.


        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')

        fan_in = input_dim * filter_size**2
        fan_out = output_dim * filter_size**2 / (stride**2)

        if mask_type is not None: # only approximately correct
            fan_in /= 2.
            fan_out /= 2.

        if he_init:
            filters_stdev = np.sqrt(4./(fan_in+fan_out))
        else: # Normalized init (Glorot & Bengio)
            filters_stdev = np.sqrt(2./(fan_in+fan_out))

        if _weights_stdev is not None:
            filter_values = uniform(
                _weights_stdev,
                (filter_size, filter_size, input_dim, output_dim)
            )
        else:
            filter_values = uniform(
                filters_stdev,
                (filter_size, filter_size, input_dim, output_dim)
            )

        # print "WARNING IGNORING GAIN"
        filter_values *= gain

        filters = lib.param(name+'.Filters', filter_values)

        if weightnorm==None:
            weightnorm = _default_weightnorm
        if weightnorm:
            norm_values = np.sqrt(np.sum(np.square(filter_values), axis=(0,1,2)))
            target_norms = lib.param(
                name + '.g',
                norm_values
            )
            with tf.name_scope('weightnorm') as scope:
                norms = tf.sqrt(tf.reduce_sum(tf.square(filters), reduction_indices=[0,1,2]))
                filters = filters * (target_norms / norms)

        if mask_type is not None:
            with tf.name_scope('filter_mask'):
                filters = filters * mask

        result = tf.nn.conv2d(
            input=inputs,
            filter=filters,
            strides=[1, 1, stride, stride],
            padding='SAME',
            data_format='NCHW'
        )

        if biases:
            _biases = lib.param(
                name+'.Biases',
                np.zeros(output_dim, dtype='float32')
            )

            result = tf.nn.bias_add(result, _biases, data_format='NCHW')


        return result

def Deconv2D(
    name,
    input_dim,
    output_dim,
    filter_size,
    inputs,
    he_init=True,
    weightnorm=None,
    biases=True,
    gain=1.,
    mask_type=None,
    ):
    """
    inputs: tensor of shape (batch size, height, width, input_dim)
    returns: tensor of shape (batch size, 2*height, 2*width, output_dim)
    """
    with tf.name_scope(name) as scope:

        if mask_type != None:
            raise Exception('Unsupported configuration')

        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')

        stride = 2
        fan_in = input_dim * filter_size**2 / (stride**2)
        fan_out = output_dim * filter_size**2

        if he_init:
            filters_stdev = np.sqrt(4./(fan_in+fan_out))
        else: # Normalized init (Glorot & Bengio)
            filters_stdev = np.sqrt(2./(fan_in+fan_out))


        if _weights_stdev is not None:
            filter_values = uniform(
                _weights_stdev,
                (filter_size, filter_size, output_dim, input_dim)
            )
        else:
            filter_values = uniform(
                filters_stdev,
                (filter_size, filter_size, output_dim, input_dim)
            )

        filter_values *= gain

        filters = lib.param(
            name+'.Filters',
            filter_values
        )

        if weightnorm==None:
            weightnorm = _default_weightnorm
        if weightnorm:
            norm_values = np.sqrt(np.sum(np.square(filter_values), axis=(0,1,3)))
            target_norms = lib.param(
                name + '.g',
                norm_values
            )
            with tf.name_scope('weightnorm') as scope:
                norms = tf.sqrt(tf.reduce_sum(tf.square(filters), reduction_indices=[0,1,3]))
                filters = filters * tf.expand_dims(target_norms / norms, 1)


        inputs = tf.transpose(inputs, [0,2,3,1], name='NCHW_to_NHWC')

        input_shape = tf.shape(inputs)
        try: # tf pre-1.0 (top) vs 1.0 (bottom)
            output_shape = tf.pack([input_shape[0], 2*input_shape[1], 2*input_shape[2], output_dim])
        except Exception as e:
            output_shape = tf.stack([input_shape[0], 2*input_shape[1], 2*input_shape[2], output_dim])

        result = tf.nn.conv2d_transpose(
            value=inputs,
            filter=filters,
            output_shape=output_shape,
            strides=[1, 2, 2, 1],
            padding='SAME'
        )

        if biases:
            _biases = lib.param(
                name+'.Biases',
                np.zeros(output_dim, dtype='float32')
            )
            result = tf.nn.bias_add(result, _biases)

        result = tf.transpose(result, [0,3,1,2], name='NHWC_to_NCHW')


        return result


def conv3d(input_, input_dim, output_dim,
           k_t=4, k_h=4, k_w=4, d_t=2, d_h=2, d_w=2, name="conv3d", padding="SAME"):
    def uniform(std_dev, size):
        return np.random.uniform(
            low=-std_dev * np.sqrt(3),
            high=std_dev * np.sqrt(3),
            size=size
        ).astype('float32')

    with tf.variable_scope(name):
        """ 
        init weights like in 
        "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
        """
        fan_in = input_dim * k_t * k_h * k_w
        fan_out = (output_dim * k_t * k_h * k_w) / (d_t * d_h * d_w)

        filters_std = np.sqrt(4. / (fan_in + fan_out))

        filter_values = uniform(
            filters_std,
            (k_t, k_h, k_w, input_dim, output_dim)
        )

        w_init = tf.Variable(filter_values, name='filters_init')
        w = tf.get_variable('filters', initializer=w_init.initialized_value())
        b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))

        # result = tf.nn.conv3d(
        #     input=input_,
        #     filter=w,
        #     strides=[1, d_t, d_h, d_w, 1],
        #     padding=padding,
        #     data_format='NDHWC'
        # )
        result = tf.nn.conv3d(
            input=input_,
            filter=w,
            strides=[1, d_t, d_h, d_w, 1],
            padding=padding
        )
        result = tf.nn.bias_add(result, b)

    return result

def Conv3D(name, input_dim, output_dim, filter_size, inputs, he_init=True, mask_type=None, stride=1, biases=True, gain=1., upFrames=None):
    """
    inputs: tensor of shape (batch size, num channels, height, width)
    mask_type: one of None, 'a', 'b'
    returns: tensor of shape (batch size, num channels, height, width)
    """
    with tf.name_scope(name) as scope:

        if mask_type is not None:
            raise NotImplemented

        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')

        fan_in = input_dim * filter_size**3 #TODO sure **3 is the right stuff here?
        fan_out = output_dim * filter_size**3 / (stride**3)

        if he_init:
            filters_stdev = np.sqrt(4./(fan_in+fan_out))
        else: # Normalized init (Glorot & Bengio)
            filters_stdev = np.sqrt(2./(fan_in+fan_out))

        # print filter_size

        output_dim = np.square(np.sqrt(output_dim)).astype('int32')
        input_dim = np.square(np.sqrt(input_dim)).astype('int32')

        # print(output_dim)


        filter_values = uniform(
            filters_stdev,
            (filter_size, filter_size, filter_size, input_dim, output_dim)
        )


        #print("filter shape: {}".format(np.shape(filter_values)))

        # print "WARNING IGNORING GAIN"
        filter_values *= gain

        filters = lib.param(name+'.Filters', filter_values)
        result = tf.nn.conv3d(
            input=inputs,
            filter=filters,
            strides=[1, 1, stride, stride, stride],
            padding='SAME'
        )


        # if biases:
        #     result = tf.transpose(result, [0, 2, 3, 4, 1], name='NCHW_to_NHWC')
        #     _biases = lib.param(
        #         name+'.Biases',
        #         np.zeros(output_dim, dtype='float32')
        #     )
        #
        #     result = tf.nn.bias_add(result, _biases)
        #     result = tf.transpose(result, [0,4,1,2,3], name='NHWC_to_NCHW')

        if biases:
            _biases = lib.param(
                name+'.Biases',
                np.zeros(output_dim, dtype='float32')
            )

            # result = tf.nn.bias_add(result, _biases, data_format='NCHW')
            result = tf.nn.bias_add(result, _biases)



        return result



def conv2d_transpose(input_, input_dim, output_shape,
                     k_h=4, k_w=4, d_h=4, d_w=4,  #L:changed from 2 to 4- dh and dw
                     name="deconv2d"):
    def uniform(std_dev, size):
        return np.random.uniform(
            low=-std_dev * np.sqrt(3),
            high=std_dev * np.sqrt(3),
            size=size
        ).astype('float32')

    with tf.variable_scope(name):
        """ 
        init weights like in 
        "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
        """

        output_dim = output_shape[-1]

        fan_in = input_dim * k_h * k_w
        fan_out = (output_dim * k_h * k_w) / (d_h * d_w)

        filters_std = np.sqrt(4. / (fan_in + fan_out))

        filter_values = uniform(
            filters_std,
            (k_h, k_w, output_dim, input_dim)
        )

        w_init = tf.Variable(filter_values, name='filter_init')
        w = tf.get_variable('filters', initializer=w_init.initialized_value())
        b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))

        result = tf.nn.conv2d_transpose(value=input_,
                                        filter=w,
                                        output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1],
                                        name=name,
                                        )

        result = tf.nn.bias_add(result, b)
        return result


def conv3d_transpose(input_, input_dim, output_shape,
                     k_h=4, k_w=4, k_d=4, d_h=2, d_w=2, d_d=2,
                     name="deconv3d"):
    def uniform(std_dev, size):
        return np.random.uniform(
            low=-std_dev * np.sqrt(3),
            high=std_dev * np.sqrt(3),
            size=size
        ).astype('float32')

    with tf.variable_scope(name):
        """ 
        init weights like in 
        "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
        """

        output_dim = output_shape[-1]

        fan_in = input_dim * k_d * k_h * k_w
        fan_out = (output_dim * k_d * k_h * k_w) / (d_d * d_h * d_w)

        filters_std = np.sqrt(4. / (fan_in + fan_out))

        filter_values = uniform(
            filters_std,
            (k_d, k_h, k_w, output_dim, input_dim)
        )

        w_init = tf.Variable(filter_values, name='filter_init')
        w = tf.get_variable('filters', initializer=w_init.initialized_value())
        b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))

        result = tf.nn.conv3d_transpose(value=input_,
                                        filter=w,
                                        output_shape=output_shape,
                                        strides=[1, d_d, d_h, d_w, 1],
                                        name=name,
                                        )

        result = tf.nn.bias_add(result, b)
        return result



def Deconv3D(
    name,
    input_dim,
    output_dim,
    filter_size,
    inputs,
    he_init=True,
    weightnorm=None,
    biases=True,
    gain=1.,
    mask_type=None,
    upFrames=None
    ):
    """
    inputs: tensor of shape (batch size, height, width, input_dim)
    returns: tensor of shape (batch size, 2*height, 2*width, output_dim)
    """
    with tf.name_scope(name) as scope:

        if mask_type != None:
            raise Exception('Unsupported configuration')

        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')

        stride = 2
        fan_in = input_dim * filter_size**3 / (stride**3)
        fan_out = output_dim * filter_size**3

        if he_init:
            filters_stdev = np.sqrt(4./(fan_in+fan_out))
        else: # Normalized init (Glorot & Bengio)
            filters_stdev = np.sqrt(2./(fan_in+fan_out))


        if _weights_stdev is not None:
            filter_values = uniform(
                _weights_stdev,
                (filter_size, filter_size, filter_size, output_dim, input_dim)
            )
        else:
            filter_values = uniform(
                filters_stdev,
                (filter_size, filter_size, filter_size, output_dim, input_dim)
            )

        filter_values *= gain

        filters = lib.param(
            name+'.Filters',
            filter_values
        )

        inputs = tf.transpose(inputs, [0, 2, 3, 4, 1], name='NCHW_to_NHWC')

        input_shape = tf.shape(inputs)
        try: # tf pre-1.0 (top) vs 1.0 (bottom)
            output_shape = tf.pack([input_shape[0], 2*input_shape[1], 2*input_shape[2], 2*input_shape[3], output_dim])
        except Exception as e:
            output_shape = tf.stack([input_shape[0], 2*input_shape[1], 2*input_shape[2], 2*input_shape[3], output_dim])

        result = tf.nn.conv3d_transpose(
            value=inputs,
            filter=filters,
            output_shape=output_shape,
            strides=[1, 2, 2, 2, 1],
            padding='SAME'
        )

        if biases:
            _biases = lib.param(
                name+'.Biases',
                np.zeros(output_dim, dtype='float32')
            )
            result = tf.nn.bias_add(result, _biases)

        #batch fr h w ch
        result = tf.transpose(result, [0,4,1,2,3], name='NHWC_to_NCHW')


        return result


def Batchnorm(name, axes, inputs, is_training=None, stats_iter=None, update_moving_stats=True, fused=True):
    if ((axes == [0,2,3]) or (axes == [0,2])) and fused==True:
        if axes==[0,2]:
            inputs = tf.expand_dims(inputs, 3)
        # Old (working but pretty slow) implementation:
        ##########

        # inputs = tf.transpose(inputs, [0,2,3,1])

        # mean, var = tf.nn.moments(inputs, [0,1,2], keep_dims=False)
        # offset = lib.param(name+'.offset', np.zeros(mean.get_shape()[-1], dtype='float32'))
        # scale = lib.param(name+'.scale', np.ones(var.get_shape()[-1], dtype='float32'))
        # result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-4)

        # return tf.transpose(result, [0,3,1,2])

        # New (super fast but untested) implementation:
        offset = lib.param(name+'.offset', np.zeros(inputs.get_shape()[1], dtype='float32'))
        scale = lib.param(name+'.scale', np.ones(inputs.get_shape()[1], dtype='float32'))

        moving_mean = lib.param(name+'.moving_mean', np.zeros(inputs.get_shape()[1], dtype='float32'), trainable=False)
        moving_variance = lib.param(name+'.moving_variance', np.ones(inputs.get_shape()[1], dtype='float32'), trainable=False)

        def _fused_batch_norm_training():
            return tf.nn.fused_batch_norm(inputs, scale, offset, epsilon=1e-5, data_format='NCHW')
        def _fused_batch_norm_inference():
            # Version which blends in the current item's statistics
            batch_size = tf.cast(tf.shape(inputs)[0], 'float32')
            mean, var = tf.nn.moments(inputs, [2,3], keep_dims=True)
            mean = ((1./batch_size)*mean) + (((batch_size-1.)/batch_size)*moving_mean)[None,:,None,None]
            var = ((1./batch_size)*var) + (((batch_size-1.)/batch_size)*moving_variance)[None,:,None,None]
            return tf.nn.batch_normalization(inputs, mean, var, offset[None,:,None,None], scale[None,:,None,None], 1e-5), mean, var

            # Standard version
            # return tf.nn.fused_batch_norm(
            #     inputs,
            #     scale,
            #     offset,
            #     epsilon=1e-2,
            #     mean=moving_mean,
            #     variance=moving_variance,
            #     is_training=False,
            #     data_format='NCHW'
            # )

        if is_training is None:
            outputs, batch_mean, batch_var = _fused_batch_norm_training()
        else:
            outputs, batch_mean, batch_var = tf.cond(is_training,
                                                       _fused_batch_norm_training,
                                                       _fused_batch_norm_inference)
            if update_moving_stats:
                no_updates = lambda: outputs
                def _force_updates():
                    """Internal function forces updates moving_vars if is_training."""
                    float_stats_iter = tf.cast(stats_iter, tf.float32)

                    update_moving_mean = tf.assign(moving_mean, ((float_stats_iter/(float_stats_iter+1))*moving_mean) + ((1/(float_stats_iter+1))*batch_mean))
                    update_moving_variance = tf.assign(moving_variance, ((float_stats_iter/(float_stats_iter+1))*moving_variance) + ((1/(float_stats_iter+1))*batch_var))

                    with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                        return tf.identity(outputs)
                outputs = tf.cond(is_training, _force_updates, no_updates)

        if axes == [0,2]:
            return outputs[:,:,:,0] # collapse last dim
        else:
            return outputs
    else:
        # raise Exception('old BN')
        # TODO we can probably use nn.fused_batch_norm here too for speedup
        mean, var = tf.nn.moments(inputs, axes, keep_dims=True)
        shape = mean.get_shape().as_list()
        if 0 not in axes:
            print "WARNING ({}): didn't find 0 in axes, but not using separate BN params for each item in batch".format(name)
            shape[0] = 1
        offset = lib.param(name+'.offset', np.zeros(shape, dtype='float32'))
        scale = lib.param(name+'.scale', np.ones(shape, dtype='float32'))
        result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-5)


        return result

def Batchnorm3D(name, axes, inputs, is_training=None, stats_iter=None, update_moving_stats=True, fused=True):
    # if ((axes == [0,2,3,4])) and fused==True:
    # if axes==[0,2]:
    #     inputs = tf.expand_dims(inputs, 3)
    # Old (working but pretty slow) implementation:
    ##########

    # inputs = tf.transpose(inputs, [0,2,3,1])

    # mean, var = tf.nn.moments(inputs, [0,1,2], keep_dims=False)
    # offset = lib.param(name+'.offset', np.zeros(mean.get_shape()[-1], dtype='float32'))
    # scale = lib.param(name+'.scale', np.ones(var.get_shape()[-1], dtype='float32'))
    # result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-4)

    # return tf.transpose(result, [0,3,1,2])

    # New (super fast but untested) implementation:
    # offset = lib.param(name+'.offset', np.zeros(inputs.get_shape()[1], dtype='float32'))
    # scale = lib.param(name+'.scale', np.ones(inputs.get_shape()[1], dtype='float32'))
    #
    # moving_mean = lib.param(name+'.moving_mean', np.zeros(inputs.get_shape()[1], dtype='float32'), trainable=False)
    # moving_variance = lib.param(name+'.moving_variance', np.ones(inputs.get_shape()[1], dtype='float32'), trainable=False)
    #
    # def _fused_batch_norm_training():
    #     return tf.nn.fused_batch_norm(inputs, scale, offset, epsilon=1e-5, data_format='NCHW')
    # def _fused_batch_norm_inference():
    #     # Version which blends in the current item's statistics
    #     batch_size = tf.cast(tf.shape(inputs)[0], 'float32')
    #     mean, var = tf.nn.moments(inputs, [2,3], keep_dims=True)
    #     mean = ((1./batch_size)*mean) + (((batch_size-1.)/batch_size)*moving_mean)[None,:,None,None]
    #     var = ((1./batch_size)*var) + (((batch_size-1.)/batch_size)*moving_variance)[None,:,None,None]
    #     return tf.nn.batch_normalization(inputs, mean, var, offset[None,:,None,None], scale[None,:,None,None], 1e-5), mean, var
    #
    #     # Standard version
    #     # return tf.nn.fused_batch_norm(
    #     #     inputs,
    #     #     scale,
    #     #     offset,
    #     #     epsilon=1e-2,
    #     #     mean=moving_mean,
    #     #     variance=moving_variance,
    #     #     is_training=False,
    #     #     data_format='NCHW'
    #     # )
    #
    # if is_training is None:
    #     outputs, batch_mean, batch_var = _fused_batch_norm_training()
    # else:
    #     outputs, batch_mean, batch_var = tf.cond(is_training,
    #                                                _fused_batch_norm_training,
    #                                                _fused_batch_norm_inference)
    #     if update_moving_stats:
    #         no_updates = lambda: outputs
    #         def _force_updates():
    #             """Internal function forces updates moving_vars if is_training."""
    #             float_stats_iter = tf.cast(stats_iter, tf.float32)
    #
    #             update_moving_mean = tf.assign(moving_mean, ((float_stats_iter/(float_stats_iter+1))*moving_mean) + ((1/(float_stats_iter+1))*batch_mean))
    #             update_moving_variance = tf.assign(moving_variance, ((float_stats_iter/(float_stats_iter+1))*moving_variance) + ((1/(float_stats_iter+1))*batch_var))
    #
    #             with tf.control_dependencies([update_moving_mean, update_moving_variance]):
    #                 return tf.identity(outputs)
    #         outputs = tf.cond(is_training, _force_updates, no_updates)
    #
    # return outputs
    # else:
    # raise Exception('old BN')
    mean, var = tf.nn.moments(inputs, axes, keep_dims=True)
    shape = mean.get_shape().as_list()
    if 0 not in axes:
        print "WARNING ({}): didn't find 0 in axes, but not using separate BN params for each item in batch".format(
            name)
        shape[0] = 1
    offset = lib.param(name + '.offset', np.zeros(shape, dtype='float32'))
    scale = lib.param(name + '.scale', np.ones(shape, dtype='float32'))
    result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-5)


    return result

def Layernorm(name, norm_axes, inputs):
    mean, var = tf.nn.moments(inputs, norm_axes, keep_dims=True)

    # Assume the 'neurons' axis is the first of norm_axes. This is the case for fully-connected and BCHW conv layers.
    n_neurons = inputs.get_shape().as_list()[norm_axes[0]]

    offset = lib.param(name+'.offset', np.zeros(n_neurons, dtype='float32'))
    scale = lib.param(name+'.scale', np.ones(n_neurons, dtype='float32'))

    # Add broadcasting dims to offset and scale (e.g. BCHW conv data)
    offset = tf.reshape(offset, [-1] + [1 for i in xrange(len(norm_axes)-1)])
    scale = tf.reshape(scale, [-1] + [1 for i in xrange(len(norm_axes)-1)])

    result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-5)

    return result


def leaky_relu(x, leak=0.2):
    """
    Code taken from https://github.com/sugyan/tf-dcgan/blob/master/dcgan.py
    """
    return tf.maximum(x, x * leak)

def dis_block2d(input, input_dim, output_dim, name, reuse=False, normalize=True):
    with tf.variable_scope(name, reuse=reuse) as vs:
        result = conv2d(input, input_dim, output_dim, name='conv2d')
        if normalize:
            result = tf.contrib.layers.layer_norm(result, reuse=reuse, scope=vs)
        result = leaky_relu(result)
    return result


def dis_block(input, input_dim, output_dim, name, reuse=False, normalize=True):
    with tf.variable_scope(name, reuse=reuse) as vs:
        result = conv3d(input, input_dim, output_dim, name='conv3d')
        if normalize:
            result = tf.contrib.layers.layer_norm(result, reuse=reuse, scope=vs)
        result = leaky_relu(result)
    return result


def linear(input_, output_size, scope=None, stddev=0.01, bias_start=0.0, with_w=False):
    """
    Code from https://github.com/wxh1996/VideoGAN-tensorflow
    """
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
