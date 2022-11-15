import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

# This class generates callback class which is called during model.fit method
class ChangeLR(keras.callbacks.Callback):
    def __init__(self, dict_):
        epochs, lrs = list(dict_.keys()), list(dict_.values())
        self.new_dict = {}
        for x in range(len(epochs)):
            if '_' in epochs[x]:
                min_, max_ = epochs[x].split('_')
                min_, max_ = int(min_), int(max_)
                for y in range(min_, max_+1):
                    self.new_dict[y] = lrs[x]
    def on_epoch_begin(self, epoch, logs):
        K.set_value(self.model.optimizer.learning_rate, self.new_dict[epoch])
        print(self.model.optimizer.learning_rate)

if __name__ == '__main__':
    ChangeLR()