import pickle
import matplotlib.pyplot as plt
from pylab import *
from sklearn.preprocessing import normalize
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
import numpy as np
import tensorflow as tf

def get_confusion_matrix():
    with open('data_best.pickle', 'rb') as f:
        data = pickle.load(f)
        return data

def produce_confusion_matrix():
    data = get_confusion_matrix()
    data = np.delete(data, (0), axis=0)
    data = np.delete(data, (0), axis=1)
    normed_matrix = normalize(data, axis=1, norm='l1')
    plt.matshow(normed_matrix)
    plt.colorbar()
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    width, height = normed_matrix.shape
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.xlabel("Ground truth")
    plt.ylabel("Predictions")
    plt.savefig('confusion_matrix.png', format='png')


def compute_mean_iou(total_cm):
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
    iou = iou[1:]
    print(iou.eval())
    return math_ops.reduce_mean(iou)

def get_IOUs():
    sess = tf.InteractiveSession()
    cm = get_confusion_matrix()
    m_iou = compute_mean_iou(cm)
    print(m_iou.eval())

get_IOUs()
#produce_confusion_matrix()
