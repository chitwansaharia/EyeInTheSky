from torch.utils.data import Dataset
import numpy as np

class SatelliteDataset(Dataset):
    """Satellite dataset."""

    def __init__(self, x_dir, y_dir, root_dir, transform=None):
        """
        Args:
            x_dir (string) : directory that contains satellite images.
            y_dir (string) : directory that contains segmentation labels.
            root_dir (string) : root directory containing the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        x_list, y_list = [], []
        x_path = "{}/{}".format(root_dir, x_dir)
        y_path = "{}/{}".format(root_dir, y_dir)

        for file in os.listdir(x_path):
            x_list.append(np.load("{}/{}".format(x_path, file)))
            y_list.append(np.load("{}/{}".format(y_path, file)))

        x_list = np.matrix(x_list)
        y_list = np.matrix(y_list)

        val_num = int(len(x_list))
        indexes = np.arange(0, len(x_list))

        np.random.shuffle(indexes)

        self.images, self.masks = x_list[indexes, :, :], y_list[indexes, :, :]

        self.length = self.images.shape[0]

    def __len__(self):
        return 10*self.length

    def __getitem__(self, idx):
        if self.transform:
            sample = self.transform(self.images[idx%self.length, :, :], self.masks[idx%self.length, :, :])
            
        return sample


class customRandomCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
             - constant: pads with a constant value, this value is specified with fill
             - edge: pads with the last value on the edge of the image
             - reflect: pads with reflection of image (without repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
             - symmetric: pads with reflection of image (repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, target):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            target = F.pad(target, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
            target = F.pad(target, (self.size[1] - target.size[0], 0), self.fill, self.padding_mode)

        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)
            target = F.pad(target, (self.size[1] - target.size[0], 0), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return (F.crop(img, i, j, h, w), F.crop(target, i, j, h, w))

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

