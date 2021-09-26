import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class ImageDataset(Dataset) :
    def __init__(self , data , label, class_size) :
        super(ImageDataset , self).__init__()
        self.data = data
        self.label = np.eye(class_size)[label-1]

    def __len__(self) :
        data_len = self.data.shape[0]
        return data_len

    def __getitem__(self , idx) :
        img_data = self.data[idx]
        img_label = self.label[idx]
        return img_data , img_label
    
class TrainTransforms :
    def __init__(self, org_size, tar_size) :
        self.org_size = org_size
        self.tar_size = tar_size
        self.transform = transforms.Compose([
            transforms.Resize(org_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(tar_size),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2))
        ])

    def __call__(self, img_tensor) :
        return self.transform(img_tensor)
    
class ValTransforms :
    def __init__(self, tar_size) :
        self.tar_size = tar_size
        self.transform = transforms.Compose([
            transforms.Resize(tar_size),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2))
        ])

    def __call__(self, img_tensor) :
        return self.transform(img_tensor)