# Author: Yash Choksi
# Date: Marc 06th 2022

# importing libraries
import os
import tensorflow as tf
def start_gpus(gpu_list):
	"""
	This function will initialize GPUs with mirrored strategy. We can still use it with the 1 GPU.
	Arguements:
	gpu_list: string format with list of gpu ids. example: "0,1"
	"""
	# Must have initialization with GPU list.
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    # This will initiate mirrorstrategy with tensorflow
    strategy = tf.distribute.MirroredStrategy()
    # As data and model has to be copied on all of GPUs.
    REPLICAS = strategy.num_replicas_in_sync
    # To copy and get data from all places we use autotune.
    AUTO = tf.data.experimental.AUTOTUNE
    print('Returning objects as strategy, replicas and auto in same order.')
    return strategy, REPLICAS, AUTO
if __name__ == '__main__':
    start_gpus()