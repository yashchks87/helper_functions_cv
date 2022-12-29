import tensorflow as tf
import pandas as pd

def get_data(**kwargs):
    imgs = [path for path in kwargs['csv_file']['images'].values.tolist()]
    labels = [path for path in kwargs['csv_file']['labels'].values.tolist()]
    shapes = [(256, 256) for x in range(len(imgs))]
    tensor = tf.data.Dataset.from_tensor_slices((imgs, labels, shapes))
    tensor = tensor.cache()
    if kwargs['repeat']:
        tensor = tensor.repeat()
    if kwargs['shuffle']:
        tensor = tensor.shuffle(1024 * REPLICAS)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        tensor = tensor.with_options(opt)
    tensor = tensor.map(get_train_imgs, num_parallel_calls = AUTO)
    if kwargs['batch']:
        tensor = tensor.batch(kwargs['batch_size'] * REPLICAS)
    if kwargs['prefetch']:
        tensor = tensor.prefetch(AUTO)
    return tensor

"""
    Dict exmaple:
    args_dict = {
        'csv_file': csv_file,
        'repeat': True,
        'shuffle': True,
        'batch': True,
        'prefetch': True,
        'batch_size': 32 or 64
    }

"""