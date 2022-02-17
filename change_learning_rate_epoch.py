# This class generates callback class which is called during model.fit method
class CallbackForSavingModelWeights(keras.callbacks.Callback):
    # Constructor takes arguement as path where weights need to be stored.
    def __init__(self, path, change_rate):
        # Boolean variable which will store value weather we like to change 
        # learning rate or not
        self.change_rate = change_rate
    def on_epoch_begin(self, epoch, logs):
        if self.change_rate == True:
            # For first 3 epochs 
            if 0 <= epoch <= 3:
                K.set_value(self.model.optimizer.learning_rate, 0.1)
            # For 4 to 20 epochs
            elif 4 <= epoch <= 20:
                K.set_value(self.model.optimizer.learning_rate, 0.01)
            # For 20 to 50 epochs
            elif 20 <= epoch <= 50:
                K.set_value(self.model.optimizer.learning_rate, 0.001)
            # For 50 to 100 epochs
            elif 50 <= epoch <= 100:
                K.set_value(self.model.optimizer.learning_rate, 0.0001)