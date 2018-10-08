import tensorflow as tf
import numpy as np

# Build a class for model
class Model():
    def __init__(self):
        pass

    def new_conv_layer(self, bottom, filter_shape, activation=tf.identity, padding='SAME', stride=1, name=None ):
        with tf.variable_scope(name):
            w = tf.get_variable(
                    "W",
                    shape=filter_shape,
                    initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.get_variable(
                    "b",
                    shape=filter_shape[-1],
                    initializer=tf.constant_initializer(0.))

            conv = tf.nn.conv2d( bottom, w, [1,stride,stride,1], padding=padding)
            bias = activation(tf.nn.bias_add(conv, b))

        return bias #relu

    def new_deconv_layer(self, bottom, filter_shape, output_shape, activation=tf.identity, padding='SAME', stride=1, name=None):
        with tf.variable_scope(name):
            W = tf.get_variable(
                    "W",
                    shape=filter_shape,
                    initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.get_variable(
                    "b",
                    shape=filter_shape[-2],
                    initializer=tf.constant_initializer(0.))
            deconv = tf.nn.conv2d_transpose( bottom, W, output_shape, [1,stride,stride,1], padding=padding)
            bias = activation(tf.nn.bias_add(deconv, b))

        return bias

    def new_fc_layer( self, bottom, output_size, name ):
        shape = bottom.get_shape().as_list()
        dim = np.prod( shape[1:] )
        x = tf.reshape( bottom, [-1, dim])
        input_size = dim

        with tf.variable_scope(name):
            w = tf.get_variable(
                    "W",
                    shape=[input_size, output_size],
                    initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.get_variable(
                    "b",
                    shape=[output_size],
                    initializer=tf.constant_initializer(0.))
            fc = tf.nn.bias_add( tf.matmul(x, w), b)

        return fc

    def channel_wise_fc_layer(self, input, name): # bottom: (7x7x512)
        _, width, height, n_feat_map = input.get_shape().as_list()
        input_reshape = tf.reshape( input, [-1, width*height, n_feat_map] )
        input_transpose = tf.transpose( input_reshape, [2,0,1] )

        with tf.variable_scope(name):
            W = tf.get_variable(
                    "W",
                    shape=[n_feat_map,width*height, width*height], # (512,49,49)
                    initializer=tf.random_normal_initializer(0., 0.005))
            output = tf.batch_matmul(input_transpose, W)

        output_transpose = tf.transpose(output, [1,2,0])
        output_reshape = tf.reshape( output_transpose, [-1, height, width, n_feat_map] )

        return output_reshape

    def leaky_relu(self, bottom, leak=0.1):
        return tf.maximum(leak*bottom, bottom)

    def batchnorm(self, bottom, is_train, epsilon=1e-8, name=None):
        bottom = tf.clip_by_value( bottom, -100., 100.)
        depth = bottom.get_shape().as_list()[-1]

        with tf.variable_scope(name):

            gamma = tf.get_variable("gamma", [depth], initializer=tf.constant_initializer(1.))
            beta  = tf.get_variable("beta" , [depth], initializer=tf.constant_initializer(0.))

            batch_mean, batch_var = tf.nn.moments(bottom, [0,1,2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)


            def update():
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            ema_apply_op = ema.apply([batch_mean, batch_var])
            ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
            mean, var = tf.cond(
                    is_train,
                    update,
                    lambda: (ema_mean, ema_var) )

            normed = tf.nn.batch_norm_with_global_normalization(bottom, mean, var, beta, gamma, epsilon, False)
        return normed

    def resize_conv_layer(self, bottom, filter_shape, resize_scale=2, activation=tf.identity, padding='SAME', stride=1, name=None):
        width = bottom.get_shape().as_list()[1]
        height = bottom.get_shape().as_list()[2]
    
        bottom = tf.image.resize_nearest_neighbor(bottom, [width*resize_scale, height*resize_scale])
        bias = self.new_conv_layer(bottom, filter_shape, stride=1, name=name )
        return bias

    def build_reconstruction( self, images, is_train ):  
        with tf.variable_scope('GEN'):
  
            # VGG 19 for Feature learning
            # 1.
            conv1_1 = self.new_conv_layer(images, [3,3,3,32], stride=1, name="conv1_1" )
            #bn1_1 = tf.nn.relu(self.batchnorm(conv1_1, is_train, name='bn1_1')) 
            conv1_1 = tf.nn.elu(conv1_1)
            conv1_2 = self.new_conv_layer(conv1_1, [3,3,32,32], stride=1, name="conv1_2" )
            #bn1_2 = tf.nn.relu(self.batchnorm(conv1_2, is_train, name='bn1_2')) 
            conv1_2 = tf.nn.elu(conv1_2)
            # Use stride convolution to replace max pooling (with padding to keep retain size 128->64)
            conv1_stride = self.new_conv_layer(conv1_2, [3,3,32,32], stride=2, name="conv1_stride")
            
            # 2.
            conv2_1 = self.new_conv_layer(conv1_stride, [3,3,32,64], stride=1, name="conv2_1" )
            #bn2_1 = tf.nn.relu(self.batchnorm(conv2_1, is_train, name='bn2_1')) 
            conv2_1 = tf.nn.elu(conv2_1)
            conv2_2 = self.new_conv_layer(conv2_1, [3,3,64, 64], stride=1, name="conv2_2" )
            #bn2_2 = tf.nn.relu(self.batchnorm(conv2_2, is_train, name='bn2_2')) 
            conv2_2 = tf.nn.elu(conv2_2)
            # Use stride convolution to replace max pooling (with padding to keep retain size 64->32)
            conv2_stride = self.new_conv_layer(conv2_2, [3,3,64,64], stride=2, name="conv2_stride")
            
            # 3.
            conv3_1 = self.new_conv_layer(conv2_stride, [3,3,64,128], stride=1, name="conv3_1" )
            #bn3_1 = tf.nn.relu(self.batchnorm(conv3_1, is_train, name='bn3_1')) 
            conv3_1 = tf.nn.elu(conv3_1)
            conv3_2 = self.new_conv_layer(conv3_1, [3,3,128, 128], stride=1, name="conv3_2" )
            #bn3_2 = tf.nn.relu(self.batchnorm(conv3_2, is_train, name='bn3_2')) 
            conv3_2 = tf.nn.elu(conv3_2)
            conv3_3 = self.new_conv_layer(conv3_2, [3,3,128,128], stride=1, name="conv3_3" )
            #bn3_3 = tf.nn.relu(self.batchnorm(conv3_3, is_train, name='bn3_3')) 
            conv3_3 = tf.nn.elu(conv3_3)
            conv3_4 = self.new_conv_layer(conv3_3, [3,3,128, 128], stride=1, name="conv3_4" )
            #bn3_4 = tf.nn.relu(self.batchnorm(conv3_4, is_train, name='bn3_4'))    
            conv3_4 = tf.nn.elu(conv3_4)
            # Use stride convolution to replace max pooling (with padding to keep retain size 32->16)
            conv3_stride = self.new_conv_layer(conv3_4, [3,3,128,128], stride=2, name="conv3_stride") # Final feature map (temporary)
            
            conv4_stride = self.new_conv_layer(conv3_stride, [3,3,128,128], stride=2, name="conv4_stride") # 16 -> 8
            conv4_stride = tf.nn.elu(conv4_stride)
            
            conv5_stride = self.new_conv_layer(conv4_stride, [3,3,128,128], stride=2, name="conv5_stride") # 8 -> 4
            conv5_stride = tf.nn.elu(conv5_stride)
            
            conv6_stride = self.new_conv_layer(conv5_stride, [3,3,128,128], stride=2, name="conv6_stride") # 4 -> 1
            conv6_stride = tf.nn.elu(conv6_stride)
 
            # 6.
            deconv5_fs = self.new_deconv_layer( conv6_stride, [3,3,128,128], conv5_stride.get_shape().as_list(), stride=2, name="deconv5_fs")
            debn5_fs = tf.nn.elu(deconv5_fs)
            
            skip5 = tf.concat([debn5_fs, conv5_stride], 3)
            channels5 = skip5.get_shape().as_list()[3]
            
            # 5.    
            deconv4_fs = self.new_deconv_layer( skip5, [3,3,128,channels5], conv4_stride.get_shape().as_list(), stride=2, name="deconv4_fs")
            debn4_fs = tf.nn.elu(deconv4_fs)
            
            skip4 = tf.concat([debn4_fs, conv4_stride], 3)
            channels4 = skip4.get_shape().as_list()[3]
            
            # 4.
            deconv3_fs = self.new_deconv_layer( skip4, [3,3,128,channels4], conv3_stride.get_shape().as_list(), stride=2, name="deconv3_fs")
            debn3_fs = tf.nn.elu(deconv3_fs)
            
            skip3 = tf.concat([debn3_fs, conv3_stride], 3)
            channels3 = skip3.get_shape().as_list()[3]
            
            # 3.
            deconv2_fs = self.new_deconv_layer( skip3, [3,3,64,channels3], conv2_stride.get_shape().as_list(), stride=2, name="deconv2_fs")
            debn2_fs = tf.nn.elu(deconv2_fs)
            
            skip2 = tf.concat([debn2_fs, conv2_stride], 3)
            channels2 = skip2.get_shape().as_list()[3]
            
            # 2.
            deconv1_fs = self.new_deconv_layer( skip2, [3,3,32,channels2], conv1_stride.get_shape().as_list(), stride=2, name="deconv1_fs")
            debn1_fs = tf.nn.elu(deconv1_fs)    
            
            skip1 = tf.concat([debn1_fs, conv1_stride], 3)
            channels1 = skip1.get_shape().as_list()[3]
            
            # 1.
            recon = self.new_deconv_layer( skip1, [3,3,3,channels1],  images.get_shape().as_list(), stride=2, name="recon") 
        return recon

    def build_adversarial(self, images, is_train, reuse=None):
        with tf.variable_scope('DIS', reuse=reuse):
    
            conv1 = self.new_conv_layer(images, [4,4,3,64], stride=2, name="conv1" )
            bn1 = self.leaky_relu(self.batchnorm(conv1, is_train, name='bn1'))
            #bn1 = tf.nn.elu(conv1) 
            conv2 = self.new_conv_layer(bn1, [4,4,64,128], stride=2, name="conv2")
            bn2 = self.leaky_relu(self.batchnorm(conv2, is_train, name='bn2'))
            #bn2 = tf.nn.elu(conv2) 
            conv3 = self.new_conv_layer(bn2, [4,4,128,256], stride=2, name="conv3")
            bn3 = self.leaky_relu(self.batchnorm(conv3, is_train, name='bn3'))
            #bn3 = tf.nn.elu(conv3) 
            conv4 = self.new_conv_layer(bn3, [4,4,256,512], stride=2, name="conv4")
            bn4 = self.leaky_relu(self.batchnorm(conv4, is_train, name='bn4'))
            #bn4 = tf.nn.elu(conv4) 


            output = self.new_fc_layer( bn4, output_size=1, name='output')

        return output[:,0]       
