import os

import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset

from pipelines import DEFAULT_SIM_CONFIG_PATH,simulation_pipeline
from utils.experiment import read_config, save_train_config
from utils.file_handling import read_detection,get_image_list

sc = read_config(DEFAULT_SIM_CONFIG_PATH)['simulator']
HEIGHT = sc['height']
WIDTH = sc['width']
DEFAULT_DATA_PATH = read_config('./config.json')['input_data_path']


#####################################################################################
# HELPER FUNCTIONS


def simulator_run_check(dataset_path, dataset_name, size, seed, params):
    """Checks if theres is dataset in the given path and if there is not
    dataset, then runs simulator pipeline to generate a dataset using the
    other parameters provided. """

    ##  Step 1a:  Remove the pass statement below and fill in the missing code.
    if not os.path.exists(dataset_path):
        return simulation_pipeline(params, size, dataset_name, seed)
        
    # Check if the given path exists. If it does not exist run the simulator
    # using given arguments: params and size.
    # Refer to the function simulation_pipeline() in pipelines.py







def get_label_array(dataset_path, detections_file_name, image_file_path):
    """
    Generate the ground truth label array using the detection file. The detection file
    contains the pixel positions of the lane for the given image_filename. 
    Using these pixel positions, generate a label array by setting values in 
    those pixel positions to 1 and remaining pixels to 0.
    """

    label = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    detection_file_path = os.path.join(dataset_path, detections_file_name)

    ## Step 1b: Fill in the missing code using the comments given below.

    # Check if the detection file exists in the path. If it does not exist, return a
    # zero valued label array.
    if not os.path.exists(detection_file_path):
        return label
        
        







    # Obtain the pixel positions of the lane for the given image from the 
    # detection file. Refer the function read_detection() in utils/file_handling.py.
    # One of the inputs for read_detection() is the image_filename. But you
    # are given image_file_path, which contains the full path. To
    # extract/split the filename from its full path, you may use the
    # appropriate function from os.path module or any other suitable string
    # module function.
    detection = read_detection(detection_file_path, os.path.split(image_file_path)[1])






    # Using the detections, generate a label array by setting those pixel
    # positions to 1. You have already done this before in the simulator test
    # notebook.
    label[detection[0], detection[1]] = 1









    return label


def convert_numpy_to_tensor(np_img):
    """Convert a numpy image to tensor image"""

    ## Step 1c: Remove the pass statement below and fill in the missing code.
    
    # Check if np_imp is 2d array or 3d array
    # Labels will be 2d arrays (in binary) and input images will be 3d arrays (in RGB)

    if np_img.ndim == 2:
        

    # For 2d array:
    # Convert to tensor using appropriate function from Pytorch.
    # Ensure that the datatype is 64 bit integer (long)
        tensor = torch.from_numpy(np_img).long()
        
        




    # For 3d array:
    # Numpy array is in H x W x C format, where H:Height, W:Width, C:Channel
    # But Pytorch tensors use C X H X W format.
    # Transpose the numpy array to tensor format.
    else:
        np_img = np.transpose(np_img, axes=(2, 0, 1))
        tensor = torch.from_numpy(np_img/255).float()




    # Convert to tensor using appropriate function from pytorch.
    # Ensure that the datatype is 32 bit float and
    # the values are normalized from range [0,255] to range [0,1]



    return tensor
    # Return the tensor.








def convert_tensor_to_numpy(tensor_img):
    """Convert the tensor image to a numpy image"""

    # # Step 1d: Remove the line below i.e. np_img = tensor_img,
    # and complete the missing code
    if tensor_img.is_cuda:
        tensor_img = tensor_img.to("cpu")

    # Numpy conversion: Pytorch has a function to do this. But the given
    # tensor may be in gpu (i.e. tensor_img.device attribute will be cuda if it is
    # in the GPU). Such tensors need to be brought back to cpu before numpy
    # conversions can be done. Refer to the appropriate function in the pytorch
    # documentation.


    np_img = (tensor_img.numpy() * 255).astype('uint8')



    # For 2d array:
    # Return the np_img array without any further action.
    
    if tensor_img.ndim == 3:
        np_img = np.transpose(np_img, axes=(1, 2, 0))


    # For 3d array:
    # np_img image is now in  C X H X W
    # transpose this array to H x W x C





    # Ensure that the datatype is 8 bit unsigned int and the values are in range
    # from 0 to 255.




    return np_img


#####################################################################################

class ImageDataset(Dataset):
    """
    Class representing Image Dataset. This class is used for both
    simulated data and true image datasets.
    If the dataset name provided does not exist, a simulated dataset will be
    created using the configuration and size provided.

    Refer: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, dataset_name, cfg_path,size=None, is_train=True,
                 dataset_parent_path=DEFAULT_DATA_PATH
                 , augmentation=None, seed=1):
        """
        Args:
            dataset_name (string):
                Folder name of the dataset.
            is_train (bool):
                Flag to used indicate training data.
                Default value: True
            dataset_parent_path (string):
                Path of the folder where the dataset folder is present.
                Default: DEFAULT_DATA_PATH from config.json
            size (int):
                Number of images to be generated by the simulator. Ignored otherwise.
            cfg_path (string):
                Config file path of your experiment
            augmentation(Augmentation object):
                Augmentation to be applied on the dataset. Augmentation is
                passed using the object from Compose class (see augmentation.py)
            seed (int):
                Seed used for random functions
                Default:1

        """
        params = read_config(cfg_path)

        self.detections_file_name = params['detections_file_name']

        self.dataset_path = os.path.join(dataset_parent_path, dataset_name)
        self.size = size
        self.augmentation = augmentation

        # Check if the directory exists
        simulator_run_check(self.dataset_path, dataset_name, self.size, seed, params)

        # Get image list and store them to a list
        dataset_folder = os.path.abspath(self.dataset_path)
        self.img_list = get_image_list(dataset_folder)
        self.size = len(self.img_list)
        # Save the dataset information in config file
        if is_train:
            save_train_config(params, self.augmentation, seed, self.dataset_path,
                              self.size)

    def __len__(self):
        """Returns length of the dataset"""
        return self.size

    def __getitem__(self, idx):
        """
        Using self.img_list and the argument value idx, return images and
        labels(if applicable) in torch tensor format.

        """
        image_file_path = self.img_list[idx]


        # # Step 1e: Delete the below line i.e. img, label = None, None and
        # complete the function implementation as indicated by the TO DO
        # comment blocks.



        #  TO DO: Read images using filename available in image_file_path
        # Hint: Use imread from scipy or any other appropriate library function.
        # img = ...

        img = imread(image_file_path)



        # Some images (usually png) have 4 channels i.e. RGBA where A is the
        # alpha channel. Since the network will use images with 3 channels.
        # Remove the 4th channel in case of RGBA images so that array has a
        # shape height x width x 3.
        img = img[:, :, :3]


        # TO DO: Obtain label array from the detection file
        # Hint: Use the helper function from step 1b.
        # label = ...

        label = get_label_array(self.dataset_path, self.detections_file_name, image_file_path)



        # Apply augmentation if applicable
        if self.augmentation is not None:
            img, label = self.augmentation((img, label))

        #  TO DO: Using the helper function from step 1, convert image and label
        #  from numpy arrays to torch tensors and return the converted image and
        #  label.
        return convert_numpy_to_tensor(img), convert_numpy_to_tensor(label)
            








