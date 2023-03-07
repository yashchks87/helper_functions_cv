import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import cv2
import PIL


# This implementation is for cv2 based images
class CustomDataset(Dataset):
    def __init__(self, data_tuples, train=True):
        paths, labels = [x[0] for x in data_tuples], [x[1] for x in data_tuples]
        self.paths, self.labels = paths, labels
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img = img.reshape(3, 256, 256)
        img = img / 255
        img = torch.tensor(img)
        img = img.type(torch.float32)
        if self.train:
            return img, self.labels[idx]
        else:
            return img

    def get_labels(self):
        return self.labels
