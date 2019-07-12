import numpy as np
import tensorflow as tf 
tf.enable_eager_execution()

layers = tf.keras.layers

class non_local_block(tf.keras.Model):
    def __init__(self, input_channel, input_shape=5, compression=1, channel_dim=-1, mode='gaussian'):
        """
            When handel video, input.shape=(batch_size, time_steps, height, width, channels), and input_shape=5
            WHen handel image, input.shape=(batch_size, height, width, channels), and input_shape=4.

            compression is set to 2 default, states halve the time/space dimension for intermediate step.
            channel_dim is set to -1 , states channel last.
        """
        super(non_local_block, self).__init__()
        self.compression = compression
        # input_shape
        self.input_channel = input_channel
        self.inputshape = input_shape
        assert self.inputshape in [4, 5]
        
        # mode
        if mode not in ["gaussian", "embedded", "dot"]:
            raise ValueError("mode must be one of [gaussian, embedded, dot]")
        self.mode = mode
        
        # channel
        channel_dim = -1   # we use channel-last

        # input_channel  (需要先知道 input_channel)
        intermediate_dim = int(input_channel / 2.)
        self.intermediate_dim = intermediate_dim

        # conv
        kernel_init = 'he_uniform'
        if mode == 'gaussian':    # only use convolution for g
            if input_shape == 4:
                self.conv_g = layers.Conv2D(intermediate_dim, kernel_size=(1,1), padding='same', use_bias=False, 
                                            kernel_initializer=kernel_init, name='conv_g')
                self.conv_o = layers.Conv2D(input_channel, kernel_size=(1,1), padding='same', use_bias=False, 
                                            kernel_initializer=kernel_init, name='conv_o')
            elif input_shape == 5:
                self.conv_g = layers.Conv3D(intermediate_dim, kernel_size=(1,1,1), padding='same', use_bias=False,
                                            kernel_initializer=kernel_init, name='conv_g')
                self.conv_o = layers.Conv3D(input_channel, kernel_size=(1,1,1), padding='same', use_bias=False, 
                                            kernel_initializer=kernel_init, name='conv_o')
            else:
                raise ValueError("input shape error")

        if mode in ["embedded", "dot"]:
            if input_shape == 4:
                self.conv_theta = layers.Conv2D(intermediate_dim, kernel_size=(1,1), padding='same', use_bias=False,
                                                kernel_initializer=kernel_init, name='conv_theta')
                self.conv_phi = layers.Conv2D(intermediate_dim, kernel_size=(1,1), padding='same', use_bias=False,
                                                kernel_initializer=kernel_init, name='conv_phi')
                self.conv_g = layers.Conv2D(intermediate_dim, kernel_size=(1,1), padding='same', use_bias=False,
                                                kernel_initializer=kernel_init, name='conv_g')
                self.conv_o = layers.Conv2D(input_channel, kernel_size=(1,1), padding='same', use_bias=False, 
                                            kernel_initializer=kernel_init, name='conv_o')
            elif input_shape == 5:
                self.conv_theta = layers.Conv3D(intermediate_dim, kernel_size=(1,1,1), padding='same', use_bias=False,
                                                kernel_initializer=kernel_init, name='conv_theta')
                self.conv_phi = layers.Conv3D(intermediate_dim, kernel_size=(1,1,1), padding='same', use_bias=False,
                                                kernel_initializer=kernel_init, name='conv_phi')
                self.conv_g = layers.Conv3D(intermediate_dim, kernel_size=(1,1,1), padding='same', use_bias=False,
                                                kernel_initializer=kernel_init, name='conv_g')
                self.conv_o = layers.Conv3D(input_channel, kernel_size=(1,1,1), padding='same', use_bias=False, 
                                            kernel_initializer=kernel_init, name='conv_o')
            else:
                raise ValueError("input shape error")

    
    def call(self, x):
        assert len(x.shape) == self.inputshape
        if self.mode == 'gaussian':
            y, w = self.predict_gaussian(x)
        if self.mode == 'embedded':
            y, w = self.predict_embedded(x)
        if self.mode == 'dot':
            y, w = self.predict_dot(x)
        return y, w

    def predict_gaussian(self, x):
        if len(x.shape) == 4:
            batch_size, dim1, dim2, channels = x.shape
        else:
            batch_size, dim1, dim2, dim3, channels = x.shape

        x_reshape = tf.reshape(x, [batch_size, -1, channels])

        # compute attention  weights
        f = tf.linalg.matmul(x_reshape, x_reshape, transpose_b=True)  # (batch_size, dim1*2*3, dim1*2*3)
        w = tf.nn.softmax(f, name="attention_weights")                # (batch_size, dim1*2*3, dim1*2*3)

        # compute outputs
        g = self.conv_g(x)
        g_reshape = tf.reshape(g, [batch_size, -1, self.intermediate_dim])   # (batch_size, dim1*2*3, inter_dim)
        y = tf.matmul(w, g_reshape)                                          # (batch_size, dim1*2*3, inter_dim)

        if len(x.shape) == 4:
            o = self.conv_o(tf.reshape(y, (batch_size, dim1, dim2, self.intermediate_dim)))
        if len(x.shape) == 5:
            o = self.conv_o(tf.reshape(y, (batch_size, dim1, dim2, dim3, self.intermediate_dim)))
        return o + x, w


    def predict_embedded(self, x):
        if len(x.shape) == 4:
            batch_size, dim1, dim2, channels = x.shape
        else:
            batch_size, dim1, dim2, dim3, channels = x.shape
        # conv
        theta = self.conv_theta(x)    # (batch_size, dim1, dim2, dim3, inter_dim)
        phi = self.conv_phi(x)        # (batch_size, dim1, dim2, dim3, inter_dim)
        
        # compute weights
        theta_reshape = tf.reshape(theta, [batch_size, -1, self.intermediate_dim])  # (batch_size, dim1*2*3, inter_dim)
        phi_reshape = tf.reshape(phi, [batch_size, -1, self.intermediate_dim])      # (batch_size, dim1*2*3, inter_dim)
        if self.compression > 1:
            phi_reshape = layers.MaxPool1D(self.compression)(phi_reshape)   # (batch_size, dim1*2*3/compression, inter_dim)

        f = tf.linalg.matmul(theta_reshape, phi_reshape, transpose_b=True)  # (batch_size, dim1*2*3, dim1*2*3/compression)
        w = tf.nn.softmax(f, name="attention_weights")   # (batch_size, dim1*2*3, dim1*2*3/compression)

        # compute output 
        g = self.conv_g(x)            # (batch_size, dim1, dim2, dim3, inter_dim)
        g_reshape = tf.reshape(g, [batch_size, -1, self.intermediate_dim])   # (batch_size, dim1*2*3, inter_dim)
        if self.compression > 1:
            g_reshape = layers.MaxPool1D(self.compression)(g_reshape)    # (batch_size, dim1*2*3/compression, inter_dim)
        o = tf.matmul(w, g_reshape)    # (batch_size, dim1*2*3, inter_dim)
        
        if len(x.shape) == 4:
            o = self.conv_o(tf.reshape(o, (batch_size, dim1, dim2, self.intermediate_dim)))
        if len(x.shape) == 5:
            o = self.conv_o(tf.reshape(o, (batch_size, dim1, dim2, dim3, self.intermediate_dim)))
        return o + x, w

    def predict_dot(self, x):
        if len(x.shape) == 4:
            batch_size, dim1, dim2, channels = x.shape
        else:
            batch_size, dim1, dim2, dim3, channels = x.shape
        # conv
        theta = self.conv_theta(x)    # (batch_size, dim1, dim2, dim3, inter_dim)
        phi = self.conv_phi(x)        # (batch_size, dim1, dim2, dim3, inter_dim)
    
        # compute weights
        theta_reshape = tf.reshape(theta, [batch_size, -1, self.intermediate_dim])  # (batch_size, dim1*2*3, inter_dim)
        phi_reshape = tf.reshape(phi, [batch_size, -1, self.intermediate_dim])      # (batch_size, dim1*2*3, inter_dim)
        w = tf.linalg.matmul(theta_reshape, phi_reshape, transpose_b=True)  # (batch_size, dim1*2*3, dim1*2*3)
        w = w / (tf.cast(w.shape[-1], tf.float32))      # normalize by N

        # compute output 
        g = self.conv_g(x)            # (batch_size, dim1, dim2, dim3, inter_dim)
        g_reshape = tf.reshape(g, [batch_size, -1, self.intermediate_dim])   # (batch_size, dim1*2*3, inter_dim)
        o = tf.matmul(w, g_reshape)    # (batch_size, dim1*2*3, inter_dim)
        
        if len(x.shape) == 4:
            o = self.conv_o(tf.reshape(o, (batch_size, dim1, dim2, self.intermediate_dim)))
        if len(x.shape) == 5:
            o = self.conv_o(tf.reshape(o, (batch_size, dim1, dim2, dim3, self.intermediate_dim)))
        return o + x, w


# if __name__ == '__main__':
#     model = non_local_block(input_channel=3, input_shape=4, compression=1, mode='embedded')
#     img = np.load("original/train_images.npy")
#     x, w = model(tf.convert_to_tensor(img[:1, 0, :4, :4, :]))
#     print(x.shape)
#     # print(img[:1, :2, :4, :4, :], "\n-------\n")
#     print(x)