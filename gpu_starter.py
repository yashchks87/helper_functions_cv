import os
import tensorflow as tf
def start_gpus(gpu_list):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    strategy = tf.distribute.MirroredStrategy()
    REPLICAS = strategy.num_replicas_in_sync
    AUTO = tf.data.experimental.AUTOTUNE
    print('Returning objects as strategy, replicas and auto in same order.')
    return strategy, REPLICAS, AUTO