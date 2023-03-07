# Author: Yash Choksi
# Date: Mar 06th 2022
# Updated: Nov 18th 2022

# importing libraries
import os
import tensorflow as tf


def start_gpus(gpu_list):
    """
    This function gives back mirrorstrategy values and gives strategy,
    replicas and auto as variable
    Args:
        gpu_list: Python list using which has gpu number is specified and you can
        use those gpu not all.
    Returns:
        strategy: Mirrorstrategy varible
        REPLICAS: How many gpus in parallel you wanna use
        AUTO: Mostly used as a part of prefetching values.
    Pissible bugs:
        Wants to remove os.environ support so figuring how we can use this only
        without using any GPU at all.
    """
    # Must have initialization with GPU list.
    physical_devices = tf.config.list_physical_devices("GPU")
    final_gpu_list = [
        physical_devices[x] for x in range(len(physical_devices)) if x in gpu_list
    ]
    # This will initiate mirrorstrategy with tensorflow
    tf.config.set_visible_devices(final_gpu_list, "GPU")
    logical_gpus = tf.config.list_logical_devices("GPU")
    strategy = tf.distribute.MirroredStrategy()
    # As data and model has to be copied on all of GPUs.
    REPLICAS = strategy.num_replicas_in_sync
    # To copy and get data from all places we use autotune.
    AUTO = tf.data.experimental.AUTOTUNE
    print("Returning objects as strategy, replicas and auto in same order.")
    return strategy, REPLICAS, AUTO


if __name__ == "__main__":
    start_gpus()
