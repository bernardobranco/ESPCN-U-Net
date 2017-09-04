import tensorflow as tf
from . import subpixel
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
import tflearn


def conv2d(inputs, num_outputs, kernel_size, scope, norm=True, activation='relu', d_format='NHWC'):
    outputs = tf.contrib.layers.conv2d(
        inputs, num_outputs, kernel_size, scope=scope,
        data_format=d_format, activation_fn=None, biases_initializer=None)
    if activation == 'relu':
        return tf.contrib.layers.batch_norm(
            outputs, decay=0.9, center=True, activation_fn=tf.nn.relu,
            updates_collections=None, epsilon=1e-5, scope=scope+'/batch_norm',
            data_format=d_format)
    elif activation == 'p_relu':
        print('YOOOO')
        return tf.contrib.layers.batch_norm(
            outputs, decay=0.9, center=True, activation_fn=tflearn.activations.prelu,
            updates_collections=None, epsilon=1e-5, scope=scope+'/batch_norm',
            data_format=d_format)

def sub_pixel_conv(inputs, scope, r, d_format='NHWC', debug=False, activation='relu'):
    # TODO: put this in subpixel function
    outputs = subpixel.PS(inputs, r)

    if activation == 'relu':
        return tf.contrib.layers.batch_norm(
            outputs, decay=0.9, activation_fn=tf.nn.relu, updates_collections=None,
            epsilon=1e-5, scope=scope+'/batch_norm', data_format=d_format)
    elif activation == 'p_relu':
        print('YOOOO')
        return tf.contrib.layers.batch_norm(
            outputs, decay=0.9, activation_fn=tflearn.activations.prelu, updates_collections=None,
            epsilon=1e-5, scope=scope+'/batch_norm', data_format=d_format)

def deconv(inputs, out_num, kernel_size, scope, d_format='NHWC'):
    outputs = tf.contrib.layers.conv2d_transpose(
        inputs, out_num, kernel_size, scope=scope, stride=[2, 2],
        data_format=d_format, activation_fn=None, biases_initializer=None)
    return tf.contrib.layers.batch_norm(
        outputs, decay=0.9, activation_fn=tf.nn.relu, updates_collections=None,
        epsilon=1e-5, scope=scope+'/batch_norm', data_format=d_format)

def pool2d(inputs, kernel_size, scope, data_format='NHWC'):
    return tf.contrib.layers.max_pool2d(
        inputs, kernel_size, scope=scope, padding='SAME',
        data_format=data_format)

def add_to_regularization_and_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name, var)
        tf.add_to_collection("reg_loss", tf.nn.l2_loss(var))


def add_activation_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name + "/activation", var)
        tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))


def add_gradient_summary(grad, var):
    if grad is not None:
        tf.summary.histogram(var.op.name + "/gradient", grad)


def compute_mean_iou(total_cm, name):
    """Compute the mean intersection-over-union via the confusion matrix."""
    sum_over_row = math_ops.to_float(math_ops.reduce_sum(total_cm, 0))
    sum_over_col = math_ops.to_float(math_ops.reduce_sum(total_cm, 1))
    cm_diag = math_ops.to_float(array_ops.diag_part(total_cm))
    denominator = sum_over_row + sum_over_col - cm_diag

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = array_ops.where(
        math_ops.greater(denominator, 0),
        denominator,
        array_ops.ones_like(denominator))
    iou = math_ops.div(cm_diag, denominator)
    print(iou.eval())
    return math_ops.reduce_mean(iou, name=name)
