# importing libraries
import tensorflow as tf

def print_tensorflow_record(file_path):
    # get file_path to dataset
    raw_dataset = tf.data.TFRecordDataset(file_path)
    # will iterate over records and print each records for sake of printing we will print only first one.
    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)
        
if '__name__' == '__main__':
    print_tensorflow_record()