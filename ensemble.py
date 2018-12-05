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


num_classes = 9


parser = argparse.ArgumentParser()
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

def return_pred_image(args, mask_dim, base_image, model_name):
	pred_shape = [base_image.shape[0], base_image.shape[1], num_classes]
	x_indices = gen_indices(base_image.shape[0], mask_dim)
	y_indices = gen_indices(base_image.shape[1], mask_dim)
	weights, pred_image = np.zeros(pred_shape), np.zeros(pred_shape)
	weight_mask = weight_mask = np.repeat(gkern(mask_dim, args.nsigma), \
				num_classes).reshape([mask_dim, mask_dim, num_classes])
	model_path = os.path.join(args.data_dir, 'saved_models', model_name)
	# Load the pretrained model
	saved_data = torch.load(model_path)
	model = unet.UNet(args.num_channels, num_classes).cuda()
	model.load_state_dict(saved_data["model_state_dict"])
	model.eval()
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
	return pred_image

def main():

	args = parser.parse_args()
	
	predicted_labels = []
	true_labels = []
	predictions_dir = 'predictions_ens'
	
	if not os.path.exists(os.path.join(args.test_data,predictions_dir)):
		os.mkdir(os.path.join(args.test_data,predictions_dir))

	for file in os.listdir(args.test_data):
		if not file.endswith('.tif'):
			continue
		print("Processing file: {}".format(file))
		image = tiff.imread(os.path.join(args.test_data, file))
		base_image = skimage.exposure.rescale_intensity(image,
						in_range='image', out_range='uint8')
		base_image = base_image/255.0

		pred_images = []

		for model_name in ['BaselineModelBS20_s0.pt','BaselineModelBS20_s1.pt','BaselineModelBS20_s2.pt','BaselineModelBS20_s3.pt']:
			print("Model name {}".format(model_name))
			mask_dim = 256
			pred_image = return_pred_image(args, mask_dim, base_image, model_name)
			pred_images.append(pred_image)
		
		# If we want to give different seeds 1/4 weight as that of individual model
		# pred_images = [np.mean(np.array(pred_images[-4:]), axis=0)]

		# for model_name in ['BaselineModel_BS20_smoothweights']:
		# 	print("Model name {}".format(model_name))
		# 	mask_dim = 256
		# 	pred_image = return_pred_image(args, mask_dim, base_image, model_name)
		# 	pred_images.append(pred_image)

		# for model_name in ['BaselineModelBS20_s0_64.pt']:
		# 	print("Model name {}".format(model_name))
		# 	mask_dim = 64
		# 	pred_image = return_pred_image(args, mask_dim, base_image, model_name)
		# 	pred_images.append(pred_image)

		for model_name in ['BaselineModelBS20_s0_128.pt']:
			print("Model name {}".format(model_name))
			mask_dim = 128
			pred_image = return_pred_image(args, mask_dim, base_image, model_name)
			pred_images.append(pred_image)

		image = tiff.imread(os.path.join(args.test_data, file))
		image = skimage.exposure.rescale_intensity(image, in_range='image', out_range='uint8')
		image_rgb_ce = skimage.exposure.equalize_adapthist(image, kernel_size=None, clip_limit=0.01, nbins=256)
		image_nir = skimage.exposure.equalize_adapthist(image[:,:,-1], kernel_size=None, clip_limit=0.01, nbins=256)
		base_image = np.concatenate([image_rgb_ce, np.expand_dims(image_nir, axis=2)], axis=2)*255
		base_image /= 255.0

		for model_name in ['BaselineModelBS20_s0_ce_flip.pt', 'BaselineModelBS20_s0_ce.pt']:
			print("Model name {}".format(model_name))
			mask_dim = 256
			pred_image = return_pred_image(args, mask_dim, base_image, model_name)
			pred_images.append(pred_image)

		pred_image = np.mean(np.array(pred_images), axis=0)

		pred_labels = np.argmax(pred_image, 2)

		color_image = np.zeros([base_image.shape[0], base_image.shape[1], 3])
		for ix,iy in np.ndindex(pred_labels.shape):
			color_image[ix, iy, :] = map_dict[pred_labels[ix, iy]]
		im = Image.fromarray(np.uint8(color_image))
		

		im.save(os.path.join(args.test_data, predictions_dir, file))
		
		if args.label_data is not None:
			predicted_labels.extend(list(pred_labels.reshape([-1])))
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
			sklearn.metrics.cohen_kappa_score(true_labels, predicted_labels)))			
		print("Confusion matrix on these images : {}".format(
			sklearn.metrics.confusion_matrix(true_labels, predicted_labels)))
		print("Precision recall class F1 : {}".format(
			sklearn.metrics.precision_recall_fscore_support(true_labels, predicted_labels)))


if __name__ == "__main__":
	main()