import argparse
import itertools
import numpy as np
from models import unet
from models.TernausNetV2.models.ternausnet2 import TernausNetV2
import os
from PIL import Image
import skimage
import scipy.stats as st
import sklearn
from sklearn import metrics
import tifffile as tiff
import torch
import pickle as pkl


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, default=None,
					help="Name of the model to be tested from")
parser.add_argument("--data-dir", default=os.getcwd(),
					help="directory to store data")
parser.add_argument("--test-data", required=True, default=None,
					help="Path to test images")
# If there are labels, also report the evaluation metrics
parser.add_argument("--label-data", default=None,
					help="Path to label images for test")
parser.add_argument("--num-channels", type=int, default=4,
                    help="number of channels of imput image to use (3 or 4)")
parser.add_argument("--nsigma", type=float, default=1.5,
                    help="Number of sigma to cover in mask")
parser.add_argument("--crop-end", action="store_true",
                    help="use crop in the end of the model")
parser.add_argument("--crop-dim", default = 256,
                    help="Dimension of crop used to train the model")
parser.add_argument("--out-dir", required=True, default=None,
                    help="Output directory for prediction images")
parser.add_argument("--pkl-dir", default = None,
                    help="Output directory for model output pkl file")
parser.add_argument("--ternaus", action="store_true",
                    help="Use ternaus network architecture for training")

"""
Returns a gaussian filter with size kernlen and sigma kernlen*nsig
Would be used for deblocking
"""
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
num_classes = 9
mean = [212.34, 291.38, 183.29, 335.85,0.12]
std = [80.87, 134.81, 114.16, 213.50, 0.33]


"""
Returns the indices to sample patches with consecutive patches sharing atleast 
50% area and also ensures all that the last patch covers max_index
"""
def gen_indices(max_index, step):
	indices = []
	index = 0
	while index + step <= max_index:
		indices.append(index)
		index += step//2
	if (max_index % (step//2) != 0) and (max_index-step > 0):
		indices.append(max_index-step)
	return indices

def return_pred_image(args, base_image, model, device):
	mask_dim = args.crop_dim
	margin = mask_dim // 10
	if args.crop_end:
		num_channels = base_image.shape[2]
		base_image = np.moveaxis(np.array([np.pad(base_image[:,:,channel], ((margin,margin),(margin,margin)), 'reflect') for channel in range(num_channels)]), 0, 2)
		x_indices = gen_indices(base_image.shape[0], mask_dim-2*margin)
		y_indices = gen_indices(base_image.shape[1], mask_dim-2*margin)
	else:
		x_indices = gen_indices(base_image.shape[0], mask_dim)
		y_indices = gen_indices(base_image.shape[1], mask_dim)

	pred_shape = [base_image.shape[0], base_image.shape[1], num_classes]
	weights, pred_image = np.zeros(pred_shape), np.zeros(pred_shape)

	# TODO maybe tune the sigma here
	weight_mask = np.repeat(gkern(mask_dim, args.nsigma), \
		num_classes).reshape([mask_dim, mask_dim, num_classes])
	if args.crop_end:
		margin = mask_dim // 10
		weight_mask[:margin,:] = 0
		weight_mask[:,:margin] = 0
		weight_mask[:,-margin:] = 0
		weight_mask[-margin:,:] = 0

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
		predictions = model(torch.Tensor(image_batch).to(device))

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
		prediction = np.mean(candidates, 0)
		
		weights[x_index:x_index+mask_dim,y_index:y_index+mask_dim,:] \
					+= weight_mask
		pred_image[x_index:x_index+mask_dim,y_index:y_index+mask_dim,:]\
					+= weight_mask*prediction

	pred_image /= weights

	if args.crop_end:
		pred_image = pred_image[margin:-margin,margin:-margin]

	return pred_image

def main():
	args = parser.parse_args()
	mask_dim = args.crop_dim
	
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # Setting device

	model_path = os.path.join(args.data_dir, 'saved_models', args.model)
	# Load the pretrained model
	saved_data = torch.load(model_path)
	if args.ternaus:
		model = TernausNetV2(num_classes = num_classes).to(device)
	else:
		model = unet.UNet(args.num_channels, num_classes).to(device)
	model.load_state_dict(saved_data["model_state_dict"])
	done_epochs = saved_data["epochs"]
	best_metric = saved_data["best_metric"]

	predicted_labels = []
	predicted_labels_images = {}
	true_labels = []

	if not os.path.exists(args.out_dir):
		os.mkdir(args.out_dir)

	for file in os.listdir(args.test_data):
		if not file.endswith('.tif'):
			continue
		print("Processing file: {}".format(file))
		
		base_image = tiff.imread(os.path.join(args.test_data, file))
		base_image = base_image.astype(float)
		num_channels = base_image.shape[2]
		for i in range(num_channels):
			base_image[:,:,i] = (base_image[:,:,i]-mean[i])/std[i]

		margin = mask_dim // 10
		if args.crop_end:
			base_image = np.moveaxis(np.array([np.pad(base_image[:,:,channel], ((margin,margin),(margin,margin)), 'reflect') for channel in range(num_channels)]), 0, 2)

		
		pred_image = return_pred_image(args, base_image, model, device)
		predicted_labels_images[file] = pred_image

		pred_image = np.argmax(pred_image, 2)

		color_image = np.zeros([base_image.shape[0], base_image.shape[1], 3])
		for ix,iy in np.ndindex(pred_image.shape):
			color_image[ix, iy, :] = map_dict[pred_image[ix, iy]]
		im = Image.fromarray(np.uint8(color_image))
		
		im.save(os.path.join(args.out_dir, file))
		
		if args.label_data is not None:
			file_name = file.split('.')[0]
			label_path = os.path.join(args.label_data, '{}.npy'.\
								format(file_name))
			predicted_labels.extend(list(pred_image.reshape([-1])))
			flat_true_label = np.expand_dims(np.load(label_path),
								axis=2).reshape(-1)
			true_labels.extend(list(flat_true_label))
	
	
	if args.label_data is not None:
		predicted_labels = np.array(predicted_labels)
		true_labels = np.array(true_labels)
		print("Accuracy on these images : {}".format(
			sklearn.metrics.accuracy_score(predicted_labels, true_labels)))
		print("Cohen kappa score on these images : {}".format(
			sklearn.metrics.cohen_kappa_score(predicted_labels, true_labels)))			
		print("Confusion matrix on these images : {}".format(
			sklearn.metrics.confusion_matrix(true_labels, predicted_labels)))
		print("Precision recall class F1 : {}".format(
			sklearn.metrics.precision_recall_fscore_support(true_labels, predicted_labels)))

	if args.pkl_dir is not None:
		with open(os.path.join(args.pkl_dir, args.model+'.pkl'),'wb') as f:
			pkl.dump(predicted_labels_images, f)

if __name__ == "__main__":
	main()