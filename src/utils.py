import tensorflow as tf
from keras.initializers import RandomNormal
from keras.optimizers import Adam


def get_std_init():
    return RandomNormal(mean=0.0, stddev=0.0001, seed=None)

def get_model_optimization():
    return Adam(lr=1e-6)

def get_model_metrics():
    return ['binary_accuracy']
    
def class_balanced_cross_entropy_loss(label, output):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    Extracted from Osvos - Caelles et. al(2017)
    url: https://github.com/scaelles/OSVOS-TensorFlow
    """

    labels = tf.cast(tf.greater(label, 0.5), tf.float32)

    num_labels_pos = tf.reduce_sum(labels)
    num_labels_neg = tf.reduce_sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = tf.cast(tf.greater_equal(output, 0), tf.float32)
    loss_val = tf.multiply(output, (labels - output_gt_zero)) - tf.log(
        1 + tf.exp(output - 2 * tf.multiply(output, output_gt_zero)))

    loss_pos = tf.reduce_sum(-tf.multiply(labels, loss_val))
    loss_neg = tf.reduce_sum(-tf.multiply(1.0 - labels, loss_val))

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    return final_loss

def get_model_loss():
    return class_balanced_cross_entropy_loss
