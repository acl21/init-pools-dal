import torch
import numpy as np
from typing import Any, Tuple
from PIL import Image

# Some of the code in this file was taken from https://github.com/gidariss/FeatureLearningRotNet/blob/master/dataloader.py
def rotate_img(img, rot):
    if rot == 0: # 0 degrees rotation
        return img
    elif rot == 90: # 90 degrees rotation
        return np.flipud(np.transpose(img, (1,0,2)))
    elif rot == 180: # 90 degrees rotation
        return np.fliplr(np.flipud(img))
    elif rot == 270: # 270 degrees rotation / or -90
        return np.transpose(np.flipud(img), (1,0,2))
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

        
class RotNetDataset(torch.utils.data.Dataset):
    """
    It is explained in the paper (https://arxiv.org/abs/1803.07728) that we get significant 
    improvement when we train the network by feeding it all the four rotated copies of an image 
    simultaneously instead of each time randomly sampling a single rotation transformation. 
    
    Therefore, at each training batch the network sees 4 times more images than the batch size.
    
    Args:
        name (string): Name of the dataset. E.g., 'CIFAR10', 'TINYIMAGENET', etc.
        dataset (Dataset, optional): PyTorch Dataset object. 
        
    """
    def __init__(self, name, dataset):
        super(RotNetDataset, self).__init__()
        self.dataset = dataset
        if name in ['CIFAR10', 'MNIST', 'SVHN']:
            self.old_samples = self.dataset.data
            self.labels = []
        else: # Tiny ImageNet
            self.old_samples = [item[0] for item in self.dataset.samples]
            self.old_samples = [np.asarray(Image.open(img).convert("RGB")) for img in self.old_samples]
        self.data, self.targets = self.create_rotations()
        self.transform = self.dataset.transform
        
    def create_rotations(self):
        imgs, labels = [], []
        for idx in range(len(self.old_samples)):
            img0 = self.old_samples[idx]
            rotated_imgs = [
                img0,
                rotate_img(img0,  90),
                rotate_img(img0, 180),
                rotate_img(img0, 270)
            ]
            rotation_labels = torch.LongTensor([0, 1, 2, 3])
            imgs.extend(rotated_imgs)
            labels.extend(rotation_labels)
        return imgs, labels
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target