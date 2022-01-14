class saveOnEveryEpoch(keras.callbacks.Callback):
	# Constructor function
    def __init__(self, path):
    	# sets the path global variable
        self.path = path
        # checks if path exists if no then first creates directory
        if os.path.exists(self.path) == False:
            os.makedirs(self.path)
    # on every epoch end it will save the weights
    def on_epoch_end(self, epoch, logs):
    	# save weights on every eopch
        self.model.save_weights(f'{self.path+str(epoch)}.h5')
        print(f'\nWeight saved for epoch {epoch}.')

from keras.callbacks import CSVLogger
# this will save csv file
CSVLogger('/content/drive/MyDrive/plant_2021/xce_res_incep/log.csv', append=True)