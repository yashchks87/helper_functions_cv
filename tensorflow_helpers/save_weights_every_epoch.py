# Author: Yash Choksi
# Date: 06th Mar 2022

import tensorflow as tf
from tensorflow import keras
import os

# This class generates callback class which is called during model.fit method
class CallbackForSavingModelWeights(keras.callbacks.Callback):
    # Constructor takes arguement as path where weights need to be stored.
    def __init__(self, path, epoch_number=0):
        # Defining self variable for path
        self.path = path
        # Creates directory if not exists 
        if os.path.exists(self.path) is False:
            os.makedirs(self.path)
        self.epoch_number = epoch_number
    # Save model weights on every epoch end
    def on_epoch_end(self, epoch, logs={}):
        # Call self model method to save weights on every epoch
        self.model.save_weights(f'{self.path}{self.epoch_number+epoch+1}.h5')

if __name__ == '__main__':
    CallbackForSavingModelWeights()