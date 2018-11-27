# python test.py --model Unet --data_dir /mnt/blossom/more/sheshansh/EyeInTheSky/data/ --test_data /mnt/blossom/more/sheshansh/EyeInTheSky/data/valid_sat/ --label_data /mnt/blossom/more/sheshansh/EyeInTheSky/data/valid_gt/

import argparse
import itertools
import numpy as np
from models import unet
import os
from PIL import Image
import skimage
import scipy.stats as st
import sklearn
from sklearn import metrics
import tifffile as tiff
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, default=None,
					help="Name of the model to be tested from")
parser.add_argument("--data_dir", default=os.getcwd(),
					help="directory to store data")
parser.add_argument("--test_data", required=True, default=None,
					help="Path to test images")
# If there are labels, also report the evaluation metrics
parser.add_argument("--label_data", default=None,
					help="Path to label images for test")
parser.add_argument("--num-channels", type=int, default=4,
                    help="number of channels of imput image to use (3 or 4)")
parser.add_argument("--nsigma", type=float, default=1.5,
                    help="Number of sigma to cover in mask")
parser.add_argument("--predictions_dir", default='predictions',
                    help="Name of predictions folder")


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

map_dict = {
	0 : (255, 255, 0),
	1 : (255, 255, 255),
	2 : (0, 125, 0),
	3 : (100, 100, 100),
	4 : (150, 80, 0),
	5 : (0, 0, 0),
	6 : (0, 0, 150),
	7 : (0, 255, 0),
	8 : (150, 150, 255)
}

def gen_indices(max_index, step):
	indices = []
	index = 0
	while index + step <= max_index:
		indices.append(index)
		index += step//2
	if (max_index % (step//2) != 0) and (max_index-step > 0):
		indices.append(max_index-step)
	return indices

def main():
	num_classes = 9
	mask_dim = 256

	args = parser.parse_args()
	model_path = os.path.join(args.data_dir, 'saved_models', args.model)
	# Load the pretrained model
	saved_data = torch.load(model_path)
	model = unet.UNet(args.num_channels, num_classes).cuda()
	model.load_state_dict(saved_data["model_state_dict"])
	model.eval()
	done_epochs = saved_data["epochs"]
	best_metric = saved_data["best_metric"]
	
	predicted_labels = []
	true_labels = []
	
	if not os.path.exists(os.path.join(args.test_data,args.predictions_dir)):
		os.mkdir(os.path.join(args.test_data,args.predictions_dir))

	for file in os.listdir(args.test_data):
		if not file.endswith('.tif'):
			continue
		print("Processing file: {}".format(file))
		image = tiff.imread(os.path.join(args.test_data, file))
		base_image = skimage.exposure.rescale_intensity(image,
						in_range='image', out_range='uint8')
		base_image = base_image/255.0

		x_indices = gen_indices(base_image.shape[0], mask_dim)
		y_indices = gen_indices(base_image.shape[1], mask_dim)

		pred_shape = [base_image.shape[0], base_image.shape[1], num_classes]
		weights, pred_image = np.zeros(pred_shape), np.zeros(pred_shape)

		# TODO maybe tune the sigma here
		weight_mask = weight_mask = np.repeat(gkern(mask_dim, args.nsigma), \
			num_classes).reshape([mask_dim, mask_dim, num_classes])

		for x_index, y_index in itertools.product(x_indices, y_indices):
			subimage = base_image[x_index:x_index+mask_dim,\
							y_index:y_index+mask_dim,:]
			image_batch = []
			for rotate_angle in [0,90,180,270]:
				rotated = skimage.transform.rotate(subimage, 
								angle=rotate_angle,
								mode='symmetric',
								preserve_range=True)
				rotated = np.moveaxis(rotated, 2, 0)
				image_batch.append(rotated)
				image_batch.append(np.flip(rotated, axis=1))
			image_batch = np.array(image_batch)
			predictions = model(torch.Tensor(image_batch).cuda())

			predictions = torch.softmax(predictions, 1)
			
			predictions = predictions.detach().cpu().numpy()
			predictions = np.moveaxis(predictions, 1, 3)
			index_image = 0
			candidates = []
			for rotate_angle in [0,90,180,270]:
				candidates.append(skimage.transform.rotate(
					predictions[index_image,:,:,:],
					angle=-rotate_angle,
					mode='symmetric',
					preserve_range=True))
				flipped_image = np.flip(skimage.transform.rotate(
					predictions[index_image+1,:,:,:],
					angle=-rotate_angle,
					mode='symmetric',
					preserve_range=True),
					axis=1)
				candidates.append(flipped_image)
				index_image += 2

			candidates = np.array(candidates)

			weights[x_index:x_index+mask_dim,y_index:y_index+mask_dim,:] \
						+= weight_mask
			pred_image[x_index:x_index+mask_dim,y_index:y_index+mask_dim,:]\
						+= weight_mask*np.mean(candidates, 0)

		pred_image /= weights
		pred_image = np.argmax(pred_image, 2)


		color_image = np.zeros([base_image.shape[0], base_image.shape[1], 3])
		for ix,iy in np.ndindex(pred_image.shape):
			color_image[ix, iy, :] = map_dict[pred_image[ix, iy]]
		im = Image.fromarray(np.uint8(color_image))
		
		im.save(os.path.join(args.test_data, args.predictions_dir, file))
		
		if args.label_data is not None:
			predicted_labels.extend(list(pred_image.reshape([-1])))
			file_name = file.split('.')[0]
			label_path = os.path.join(args.label_data, '{}.npy'.\
								format(file_name))
			flat_true_label = np.expand_dims(np.load(label_path),
								axis=2).reshape(-1)
			true_labels.extend(list(flat_true_label))
	
	
	if args.label_data is not None:
		predicted_labels = np.array(predicted_labels)
		true_labels = np.array(true_labels)
		print("Cohen kappa score on these images : {}".format(
			sklearn.metrics.cohen_kappa_score(predicted_labels, true_labels)))			


if __name__ == "__main__":
	main()