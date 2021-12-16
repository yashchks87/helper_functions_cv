# This class generates callback class which is called during model.fit method
class CallbackForSavingModelWeights(keras.callbacks.Callback):
    # Constructor takes arguement as path where weights need to be stored.
    def __init__(self, path):
        # Defining self variable for path
        self.path = path
        # Creates directory if not exists 
        if os.path.exists(self.path) is False:
            os.makedirs(self.path)
    # Save model weights on every epoch end
    def on_epoch_end(self, epoch, logs={}):
        # Call self model method to save weights on every epoch
        self.model.save_weights(f'{self.path}{epoch+1}.h5')