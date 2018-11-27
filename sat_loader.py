from torch.utils.data import Dataset
import numpy as np
import tifffile as tiff
from random import shuffle
import skimage
from skimage import transform
import numpy as np
import os
import random


class SatelliteDataset(Dataset):
    """
    Satellite dataset loader.
    Only to be used during training and validation
    """

    def __init__(self, x_dir, y_dir, root_dir, crop_dim, num_channels):
        """
        Args:
            x_dir (string) : directory that contains satellite images.
            y_dir (string) : directory that contains segmentation labels.
            root_dir (string) : root directory containing the dataset.
        """
        self.crop_dim = crop_dim
        self.num_channels = num_channels
        x_list, y_list = [], []
        x_path = "{}/{}".format(root_dir, x_dir)
        y_path = "{}/{}".format(root_dir, y_dir)

        for file in os.listdir(x_path):
            file_name = file.split('.')
            if len(file_name) != 2 or file_name[1] != 'tif':
                continue
            file_name = file_name[0]
            image = tiff.imread("{}/{}".format(x_path, file))
            x_list.append(skimage.exposure.rescale_intensity(image, in_range='image', out_range='uint8'))
            y_list.append(np.expand_dims(np.load("{}/{}".format(y_path, '{}.npy'.format(file_name))), axis=2))

        self.images, self.masks = x_list, y_list
        self.length = len(self.images)

        # Custom Transforms
        self.rcrop = customRandomCrop(crop_dim)

    def __len__(self):
        return 100*self.length

    def __getitem__(self, idx):
        sample_x, sample_y = self.images[idx%self.length], self.masks[idx%self.length]
        combined = np.concatenate([sample_x, sample_y], axis=2)
        cropped_combined = self.rcrop(combined)
        assert cropped_combined.shape[0] == self.crop_dim
        assert cropped_combined.shape[1] == self.crop_dim

        _rotate_angle = np.random.uniform(0, 360)

        rotated_combined = skimage.transform.rotate(cropped_combined, angle=_rotate_angle, mode='symmetric',preserve_range=True)
        final_image = np.transpose(rotated_combined[:,:,:self.num_channels], (2,0,1))
        return self.normalize(final_image.astype(np.float32)), rotated_combined[:,:,-1]

    def normalize(self, image):
        return image/255.0


class customRandomCrop(object):
    def __init__(self, size):
        self.size = (int(size), int(size))

    @staticmethod
    def get_params(img, output_size):
        h, w = img.shape[0], img.shape[1]
        th, tw, c = output_size[0], output_size[1], img.shape[2]
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        assert self.size[0] <= img.shape[0]
        assert self.size[1] <= img.shape[1]
        i, j, h, w = self.get_params(img, self.size)
        return img[i:i+self.size[0], j:j+self.size[1], :]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

