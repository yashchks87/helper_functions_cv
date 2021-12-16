# This class inherit keras callback class and it does is it calculate false positive rate 
# on every epoch for training and validation set
class CallbackForFalsePositive(keras.callbacks.Callback):
    # Constructor function takes traing and validation tfrec path
    def __init__(self, train_tfrec, val_tfrec):
        # Calls the getData function to evaluate from tfrec file
        train = getData(train_tfrec_utk, augment=False, shuffle=False, repeat=False)
        # Unbatch the train dataset
        train = train.unbatch()
        # It will only return labels as it droppped images
        train_img = train.map(lambda img, label: img)
        # Again creating batches
        train_img = train_img.batch(16 * REPLICAS)
        train_label = train.map(lambda img, label: label)
        # Converting true(ground truth labels) to numpy to get for comparison
        # Almost similar process for validation dataset too.
        train_label = tfds.as_numpy(train_label)
        train_label = [x for x in train_label]
        val = getData(val_tfrec_utk, augment=False, shuffle=False, repeat=False)
        val = val.unbatch()
        val_img = val.map(lambda img, label: img)
        val_img = val_img.batch(16 * REPLICAS)
        val_label = val.map(lambda img, label: label)
        val_label = tfds.as_numpy(val_label)
        val_label = [x for x in val_label]
        # Create state variables for accessing in other methods
        self.train_data = train_img
        self.train_g_t = train_label
        self.val_data = val_img
        self.val_g_t = val_label
        self.train_false_positive_rates, self.val_false_positive_rates = [], []
    # This method will be called on epoch end
    def on_epoch_end(self, epoch, logs={}):
        # Predicts train and val
        train_predicted = self.model.predict(self.train_data)
        val_predicted = self.model.predict(self.val_data)
        # Calls this function which will calculate false positive rate
        train_f_p = test_false_positive_metric(self.train_g_t, train_predicted)
        val_f_p = test_false_positive_metric(self.val_g_t, val_predicted)
        # Printing.....
        print(f'\nTrain false positive rate: {train_f_p.numpy()}')    
        print(f'\nVal false positive rate: {val_f_p.numpy()}')      