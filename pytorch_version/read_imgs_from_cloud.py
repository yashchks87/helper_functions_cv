class CreateCustomDataset(Dataset):
    def __init__(self, data_tuples, size_params, train=True):
        self.data_tuples = data_tuples
        self.image_paths, self.labels = [x[0] for x in data_tuples], [
            x[1] for x in data_tuples
        ]
        self.size_params = size_params
        self.train = train

    def __len__(self):
        return len(self.data_tuples)

    def __getitem__(self, idx):
        # It will get images from direct public url paths
        img = requests.get(self.image_paths[idx])
        # It will convert images from bytes read to actual numpy array of images
        # as it's opencv they are in BGR format
        img = cv2.imdecode(np.frombuffer(io.BytesIO(img.content).read(), np.uint8), 1)
        # Simple BGR to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # resizing of images
        img = cv2.resize(img, self.size_params)
        # normalizing of images by putting everything in between 0 and 1
        img = img / 255
        # reshaping of images in accordance to pytorch tensor
        img = img.reshape(3, self.size_params[0], self.size_params[1])
        # convert numpy array to tensor
        img = torch.tensor(img)
        # very essential to cast to float 32 becuase models accept only 32 floats..
        img = img.type(torch.float32)
        if self.train:
            return img, self.labels[idx]
        else:
            return img

    def get_labels(self):
        return self.labels
