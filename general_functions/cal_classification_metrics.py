import numpy as np
import numpy


class GetMatrics:
    def __init__(self, truth, preds, positive=1, negative=0):
        # type checking in case user has given python list
        # if type(truth) != numpy.ndarray:
        #     truth = np.array(truth)
        # if type(preds) != numpy.ndarray:
        #     truth = np.array(preds)
        self.truth, self.preds = truth, preds
        # positive can be anything depends on case specific but
        # by default we take 1
        self.positive, self.negative = positive, negative
        # This will calculate all true positives
        self.TP = np.sum(
            np.logical_and(self.truth == self.positive, self.preds == self.positive)
        )
        # This will calculate all true negatives
        self.TN = np.sum(
            np.logical_and(self.truth == self.negative, self.preds == self.negative)
        )
        # This will calculate all false positives
        self.FP = np.sum(
            np.logical_and(self.truth == self.negative, self.preds == self.positive)
        )
        # This will calculate all false negatives
        self.FN = np.sum(
            np.logical_and(self.truth == self.positive, self.preds == self.negative)
        )

        print(f"TP: {self.TP}")
        print(f"FP: {self.FP}")
        print(f"TN: {self.TN}")
        print(f"FN: {self.FN}")

    def get_precision(self):
        return self.TP / (self.TP + self.FP)

    def get_recall(self):
        return self.TP / (self.TP + self.FN)

    def get_confusion_matrix(self):
        print("Order as 1 and 0...")
        return np.array([[self.TP, self.FN], [self.FP, self.TN]])


if __name__ == "__main__":
    GetMatrics()
