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
import pickle
import copy

num_classes = 9
parser = argparse.ArgumentParser()
parser.add_argument("--pred-dir", required=True, default=None,
					help="Path to save the predicted masks")
parser.add_argument("--label-data", default=None,
					help="Path to label images for test")

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

def main():
	args = parser.parse_args()
	predictions_dir = 'predictions_ens'
	pickle_files = ["no_rescale_weighted0.2_pretrained.pt.pkl", "ternauspretrain.pt.pkl", "no_rescale_soil_pretrained.pt.pkl"]

	predictions_list = []
	predicted_labels = []
	true_labels = []
	for file in pickle_files:
		with open('pkl/{}'.format(file),'rb') as f:
			u = pickle._Unpickler(f)
			u.encoding = 'latin1'
			p = u.load()
			predictions_list.append(p)
	images = p.keys()

	if not os.path.exists(os.path.join(args.pred_dir,predictions_dir)):
		os.mkdir(os.path.join(args.pred_dir,predictions_dir))

	for file in images:
		pred_images = []
		print("Processing file: {}".format(file))
		for index in range(len(predictions_list)):
			pred_images.append(predictions_list[index][file])

		pred_image = np.mean(np.array(pred_images), axis=0)
		pred_labels = np.argmax(pred_image, 2)

		# Save Mask as an image after mapping labels to RGB colors
		color_image = np.zeros([predictions_list[0][file].shape[0], predictions_list[0][file].shape[1], 3])
		for ix,iy in np.ndindex(pred_labels.shape):
			color_image[ix, iy, :] = map_dict[pred_labels[ix, iy]]
		im = Image.fromarray(np.uint8(color_image))

		im.save(os.path.join(args.pred_dir, predictions_dir, file))

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
		print("Accuracy on these images : {}".format(
			sklearn.metrics.accuracy_score(predicted_labels, true_labels)))
		print("Precision recall class F1 : {}".format(
			sklearn.metrics.precision_recall_fscore_support(true_labels, predicted_labels)))
if __name__ == "__main__":
	main()