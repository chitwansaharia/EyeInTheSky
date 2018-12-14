# Eye In The Sky Challenge
## Inter IIT Tech Meet 2018 

The following codebase has been created as a part of Inter IIT Tech Meet 2018 Challenge Eye In The Sky satellite segmentation challenge. The code can be used for general satellite segmentation tasks as well. 

### Dataset Preparation : 
* Prepare the training and validation split, and store the masks and satellite images in the corresponding folders. 
* Change the dataset path in `train.py`
  * `root_dir` : root directory that contains the entire dataset.
  * `train_y_dir` : path of directory containing training masks relative to `root_dir`
  * `train_x_dir` : path of directory containing training images relative to `root_dir`
  * `val_y_dir` : path of directory containing validation masks relative to `root_dir`
  * `val_x_dir` : path of directory containing validation images relative to `root_dir`
* The satellite masks should be stored as discrete labels (0-9). For this particular task, we use the following RGB colour encoding: `{(255, 255, 0) : 0, (255, 255, 255): 1, (0, 125, 0): 2, (100, 100, 100): 3, (150, 80, 0): 4, (0, 0, 0): 5, (0, 0, 150): 6, (0, 255, 0): 7, (150, 150, 255): 8}`


### Model Architectures : 
`models/` contains the different model architectures, currently supporting **UNet** (Inspired from [here](https://github.com/milesial/Pytorch-UNet)) and **TernausNet** (Inspired from [here](https://github.com/ternaus/TernausNet))

### Training : 
`train.py` can be used to train the model once, the dataset is prepared. 
The most basic command to run this is : 
`python3 train.py --model <model name>`

Main command line arguments that can be used with `train.py`:
  * `--model` : Model name to be used for saving the model. The model is saved in `saved_models/` 
  * `--data_dir` : Parent directory to store the saved models, and tensorboard loggings
  * `--batch-size` : Batch size to be used (256x256 crop with batch size of 20 takes ~ 5GB of GPU memory)
  * `--ternaus` : Use Ternaus Net as the model architecture 
  * `--resume` : Use an existing model (name specified with `--model`) (The saved model must exist)
  * `--num-channels` : Number of channels to use for satellite images. Currently, we use 4 channels (R,G,B,NIR). But the        `sat_loader.py` contains other channel calculations (NDWI, NDVI, SAVI), and they can be appended using `--num-channels`. 
  * `--crop-dim` : Cropping dimension to use while random-cropping the images
  * `--log-interval` : The interval at which logging should be done 
  * `--train-per-class` : To be used when we want to train the model just for binary segmentation. The class number (to be used as the positive class) can be specified by `--class-number`, and the `--class-weight` can be used to specify the weight applied to that class (can be used when the distribution is skewed). The weight should be relative to 1
  * `--contrast-enhance` : Contrast enhance the training images 
  * `--rescale-intensity` : Rescale the intensity of the training images to `uint-8`. In this task, the satellite images are of type `uint16`
  * `--gaussian-blur` : Perform Gaussian Blur on the One-Hot segmentation masks (Inspired from [here](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Kuo_Deep_Aggregation_Net_CVPR_2018_paper.pdf))

**Change the mean and variance of different channels in** `sat_loader.py` **The present mean and variance can be used when NOT using** `--rescale-intensity` and `--contrast-enhance`. 

### Dataset Augmentation
We first randomly crop a section from the image (size specified with `--crop-dim`, and then perform random rotation of the cropped section (0-360 degrees), and then perform random flip (horizontal, vertical or no flip). When performing rotation, reflection is used to maintain the dimension of the image. 

### Testing
`test.py` can be used to test the model once, some model is trained. 
The most basic command to run this is : 
`python3 test.py --model <model name>.pt --test-data <directory containing validation images> --out-dir <directory to save output images>`
Add all the other non-default arguments used while training as well.
Additionally, test.py supports these arguments
  * `--pkl-dir` : The directory to store the pickle file for model outputs as a dictionary. Pickle would be used for ensembling. If not present, the pickle file is not saved.
  * `--nsigma` : While deblocking the standard deviation for gaussian mask as a multiple of mask dimension.
  
`ensemble.py` can be used to test the ensemble of various models. To use this, do the following:
   * Using `test.py` generate the prediction pickle files for all the models separately. 
   * Change the list `pickle_files` in `ensemble.py` with the required predictions files you want to take ensemble of.
   * run `python3 ensemble.py --pred-dir <directory to store predicted masks> --label-data <path to ground truth>`
   * If path to ground truth is provided, then it evaluates the model at the end. 
   
