from abc import ABC

import tensorflow as tf
import tensorflow.keras.backend as K


class FmLayer(tf.keras.layers.Layer):
    def __init__(self, k, w_reg, v_reg):
        super().__init__()
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

        self.w0, self.w, self.v = None, None, None

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True,)
        self.w = self.add_weight(name='w', shape=(input_shape[-1], 1),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 # regularizer=tf.keras.regularizers.l2(self.v_reg)
                                 )

        self.v = self.add_weight(name='v', shape=(input_shape[-1], self.k),
                                 initializer=tf.random_normal_initializer(0, 0.2),
                                 trainable=True,
                                 # regularizer=tf.keras.regularizers.l2(self.v_reg)
                                 )

        # normalvariate(0, 0.2) * np.ones((feature_num, self.dimension_num))

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

        linear_part = tf.matmul(inputs, self.w) + self.w0   # shape:(batch_size, 1)
        inter_part1 = tf.pow(tf.matmul(inputs, self.v), 2)  # shape:(batch_size, self.k)
        inter_part2 = tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2))   # shape:(batch_size, self.k)
        inter_part = 0.5 * tf.reduce_sum(inter_part1 - inter_part2, axis=-1, keepdims=True)   # shape:(batch_size, 1)

        output = linear_part + inter_part
        return tf.nn.sigmoid(output)


class FM(tf.keras.Model, ABC):
    def __init__(self, k, w_reg=1e-4, v_reg=1e-4):
        super().__init__()
        self.fm = FmLayer(k, w_reg, v_reg)

    def call(self, inputs, training=None, mask=None):
        output = self.fm(inputs)
        return output
