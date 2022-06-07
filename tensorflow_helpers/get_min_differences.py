# This function will calculates the minimum differences from all epoch numbes and get the 
# best from loss perspective. It can helps us in getting least overfitting of the model.
def calculate_min_differences(history_object):
    # Get history object
    hist = history_object.history
    # Get loss and val_loss attributes, need to change it something else is in the history
    # dict
    loss, val_loss = hist['loss'], hist['val_loss']
    # Python list to store all difference of losses to get clear idea
    abs_differences = []
    # List to go over all epoch losses
    for x in range(len(loss)):
        temp = abs(loss[x] - val_loss[x])
        abs_differences.append(temp)
    # Get min differences so we can get best 
    min_diff = min(abs_differences)
    # Get epoch number
    index_pos = abs_differences.index(min_diff)
    print(f'Minimum difference is: {min_diff} and it is at epoch number: {index_pos + 1}.')