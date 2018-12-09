from torch.utils.data import Dataset
import numpy as np
import tifffile as tiff
from random import shuffle
import skimage
from skimage import transform
import numpy as np
import os
import random
import torch
import scipy
from scipy import ndimage

class SatelliteDataset(Dataset):
    """
    Satellite dataset loader.
    Only to be used during training and validation
    """

    def __init__(self, x_dir, y_dir, root_dir, crop_dim, num_channels, contrast_enhance, gaussian_blur):
        """
        Args:
            x_dir (string) : directory that contains satellite images.
            y_dir (string) : directory that contains segmentation labels.
            root_dir (string) : root directory containing the dataset.
        """
        self.crop_dim = crop_dim
        self.num_channels = num_channels
        self.gaussian_blur = gaussian_blur

        x_list, y_list = [], []
        x_path = "{}/{}".format(root_dir, x_dir)
        y_path = "{}/{}".format(root_dir, y_dir)

        for file in os.listdir(x_path):
            file_name = file.split('.')
            if len(file_name) != 2 or file_name[1] != 'tif':
                continue
            file_name = file_name[0]
            image = tiff.imread("{}/{}".format(x_path, file))

            # Rescales Intensity
            # image = skimage.exposure.rescale_intensity(image, in_range='image', out_range='uint8')

            if contrast_enhance:
                image_rgb_ce = skimage.exposure.equalize_adapthist(image, kernel_size=None, clip_limit=0.01, nbins=256)
                image_nir = skimage.exposure.equalize_adapthist(image[:,:,-1], kernel_size=None, clip_limit=0.01, nbins=256)
                image = np.concatenate([image_rgb_ce, np.expand_dims(image_nir, axis=2)], axis=2)*255

            x_list.append(image)
            y_list.append(np.expand_dims(np.load("{}/{}".format(y_path, '{}.npy'.format(file_name))), axis=2))
        self.mean = [212.34, 291.38, 183.29, 335.85,0.12]
        self.std = [80.87, 134.81, 114.16, 213.50, 0.33]

        self.images, self.masks = x_list, y_list
        self.length = len(self.images)

        # Custom Transforms
        self.rcrop = customRandomCrop(crop_dim)

    def __len__(self):
        return 200*self.length

    def __getitem__(self, idx):
        sample_x, sample_y = self.images[idx%self.length], self.masks[idx%self.length]
        combined = np.concatenate([sample_x, sample_y], axis=2)
        cropped_combined = self.rcrop(combined)
        assert cropped_combined.shape[0] == self.crop_dim
        assert cropped_combined.shape[1] == self.crop_dim

        _rotate_angle = np.random.uniform(0, 360)

        rotated_combined = skimage.transform.rotate(cropped_combined, angle=_rotate_angle, mode='symmetric',preserve_range=True)

        flip = np.random.choice(3)
        if flip < 2:
            rotated_combined = np.flip(rotated_combined, axis=flip).copy()

        # ndwi_channel = (rotated_combined[:,:,1] - rotated_combined[:,:,3])/(rotated_combined[:,:,1] + rotated_combined[:,:,3])
        # ndwi_channel[ndwi_channel < 0] = 0
        #
        # final_image = self.normalize(rotated_combined[:,:,:min(self.num_channels, 4)].astype(np.float32))
        #
        # if self.num_channels >= 5:
        #     final_image = np.concatenate([final_image, np.expand_dims(nir_channel,axis=2)], axis=2).astype(np.float32)
        #
        # if self.num_channels == 6:
        #     savi_channel = (1.5)*(final_image[:,:,3] - final_image[:,:,0])/(final_image[:,:,3] + final_image[:,:,0] + 0.5)
        #     final_image = np.concatenate([final_image, np.expand_dims(savi_channel,axis=2)], axis=2).astype(np.float32)

        final_image = rotated_combined[:,:,:min(self.num_channels, 4)].astype(np.float32)

        ndvi_channel = (final_image[:,:,3] - final_image[:,:,0] + 1e-10)/(final_image[:,:,3] + final_image[:,:,0] + 1e-10)
        ndvi_channel = np.expand_dims(ndvi_channel, axis=2)

        if self.num_channels == 5:
            final_image = np.concatenate((final_image, ndvi_channel), axis=2)

        final_image = torch.tensor(np.transpose(final_image, (2,0,1)))
        self.normalize(final_image)

        if self.gaussian_blur:
            mask = self.gaussian_smooth(torch.tensor(rotated_combined[:,:,-1]))
        else:
            mask = rotated_combined[:,:,-1]

        return final_image.float(), torch.tensor(mask), torch.tensor(rotated_combined[:,:,-1])

    def normalize(self, image):
        for t, m, s in zip(image, self.mean[:self.num_channels], self.std[:self.num_channels]):
            t.sub_(m).div_(s)

    def gaussian_smooth(self, mask):
        one_hot_mask = torch.zeros([9, self.crop_dim, self.crop_dim])
        one_hot_mask.zero_()
        one_hot_mask.scatter_(0, mask.long().unsqueeze(0), 1.)
        nmask = scipy.ndimage.filters.gaussian_filter(one_hot_mask.numpy(), sigma=[0, 5, 5])
        mask_sum = np.sum(nmask, axis=0)
        return np.divide(nmask, np.expand_dims(mask_sum,0))


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

