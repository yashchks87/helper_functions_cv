# Author: Yash Choksi
# Date: Mar 06th 2022


import tensorflow as tf
def tpu_start(device_name):
    device = device_name
    # TPU and GPU mechanism
    if device == 'TPU':
        print('Connecting to TPU')
        try:
            # Detects weather system has TPU.
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            print('Running on TPU!', tpu.master())
        except ValueError:
            print('Could not connect to TPU.')
            tpu = None
        if tpu:
            try:
                print('Starting TPU')
                # Connects to the TPU cluster
                tf.config.experimental_connect_to_cluster(tpu)
                # Starts TPU cluster
                tf.tpu.experimental.initialize_tpu_system(tpu)
                # Object gives TPU cluster with 7 replicas
                strategy = tf.distribute.experimental.TPUStrategy(tpu)
                print('TPU started.')
            except:
                print('Failed to start TPU.')
        else:
            device = 'GPU'
    # If no TPU then...
    if device != 'TPU':
        print('Using single GPU and CPU.')
        strategy = tf.distribute.get_strategy()
    
    if device == 'GPU':
        print(f'Number of GPU available: {len(tf.config.experimental.list_physical_devices("GPU"))}')

    AUTO = tf.data.experimental.AUTOTUNE
    REPLICAS = strategy.num_replicas_in_sync
    print(f'Replicas: {REPLICAS}')
    print('Return auto, replica and strategy object in the same order.')
    return AUTO, REPLICAS, strategy

if '__name__' == '__main__':
    tpu_start()
    
