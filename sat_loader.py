from torch.utils.data import Dataset
import numpy as np
import tifffile as tiff
from random import shuffle
import skimage
from skimage import transform
import numpy as np

class SatelliteDataset(Dataset):
    """Satellite dataset."""

    def __init__(self, x_dir, y_dir, root_dir, crop_dim):
        """
        Args:
            x_dir (string) : directory that contains satellite images.
            y_dir (string) : directory that contains segmentation labels.
            root_dir (string) : root directory containing the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.crop_dim = crop_dim
        x_list, y_list = [], []
        x_path = "{}/{}".format(root_dir, x_dir)
        y_path = "{}/{}".format(root_dir, y_dir)

        for file in os.listdir(x_path):
            image = tiff.imread("{}/{}".format(x_path, file))
            x_list.append(skimage.exposure.rescale_intensity(image, in_range='image', out_range='uint8'))
            y_list.append(tiff.imread("{}/{}".format(y_path, file)))

        x_list, y_list = np.matrix(x_list), np.matrix(y_list)
        indexes = np.arange(0, x_list.shape[0])
        np.random.shuffle(indexes)
        self.images, self.masks = x_list[indexes, :, :, :], y_list[indexes, :, :, :]
        self.length = self.images.shape[0]

        # Custom Transforms
        self.rcrop = customRandomCrop(crop_dim)

    def __len__(self):
        return 100*self.length

    def __getitem__(self, idx):
        sample_x, sample_y = self.images[idx%self.length, :, :, :], self.masks[idx%self.length, :, :, :]
        combined = np.concatenate([sample_x, sample_y], axis=2)
        cropped_combined = self.rcrop(combined)

        assert cropped_combined.shape[0] == self.crop_dim
        assert cropped_combined.shape[1] == self.crop_dim

        _rotate_angle = np.random.uniform(0, 360)

        rotated_combined = skimage.transform.rotate(cropped_combined, angle=3, mode='symmetric',preserve_range=True)
        return rotated_combined[:,:,:-1], rotated_combined[:,:,-1]


class customRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        w, h = img.shape[0], img.shape[1]
        th, tw, c = output_size, img.shape[2]
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

