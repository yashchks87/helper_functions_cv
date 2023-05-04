from torch.utils.tensorboard import SummaryWriter
import os

class TBCallBack():
    def __init__(self, dir_path, folder_name):
        self.dir_path = dir_path
        self.folder_name = folder_name
        self.check_paths()
        self.init_writer()

    def init_writer(self):
        self.writer = SummaryWriter(f'{self.dir_path}{self.folder_name}')
        print(f'Writer object is poiniting to {self.dir_path}{self.folder_name}.')        

    def check_paths(self):
        temp_path = f'{self.dir_path}{self.folder_name}/'
        if os.path.exists(temp_path) == True:
            os.rmdir(temp_path)

    def save(self, epoch, phase, iou, loss):
        if phase == 'train':
            self.writer.add_scalars('loss', {'train': iou}, epoch)
            self.writer.add_scalars('iou', {'train': loss}, epoch)
        else:
            self.writer.add_scalars('loss', {'val': iou}, epoch)
            self.writer.add_scalars('iou', {'val': loss}, epoch)