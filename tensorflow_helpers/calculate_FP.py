# Author: Yash Choksi
# Date: Mar 06th 2022


# This function is pure tensorflow function. Takes input as y_ground truth and predicted
def test_false_positive_metric(y_true, y_pred):
    # Fix the shape of the predicted array
    y_pred = [x[0] for x in y_pred]
    # From python list(or numpy array) to tf constant tensors
    y_true = tf.constant(y_true)
    y_pred = tf.constant(y_pred)
    # Round preds so 0.5 probability will be maintained
    y_pred = tf.math.round(y_pred)
    # Calculate true negative and false positives
    TN = tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred, 0))
    FP = tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred, 1))
    TN_counts = tf.reduce_sum(tf.cast(TN, tf.int32))
    FP_counts = tf.reduce_sum(tf.cast(FP, tf.int32))
    adder = tf.math.add(FP_counts, TN_counts)
    # Calculate false positive rate
    fp_rate = tf.math.divide(FP_counts, adder)
    return fp_rate
