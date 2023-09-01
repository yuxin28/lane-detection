"""Coloring functions used to create a 3-channel numpy array representing an
RGB image from a layer represented by a bitmask. The output numpy array
should support 8 bit color depth (i.e. its data type is uint8).

@author: Sebastian Lotter <sebastian.g.lotter@fau.de>
"""
import numpy as np
import random
from skimage.io import imread
from skimage.transform import resize
from utils.file_handling import get_image_list
from matplotlib import pyplot as plt


def color_w_constant_color(fmask, color):
    """
    Color whole layer with a constant color.

    Input:
    fmask -- binary mask indicating the shape that is to be coloured
    color -- (r,g,b) tuple

    Output:
    numpy array of dimensions (x,y,3) representing 3-channel image
    """


    ## Step 2

    # Initialize empty numpy array with datatype 8-bit unsigned integer using
    # variable name 'img'.
    # img = ...
    img = np.tile(np.array(color, dtype = np.uint8), (fmask.shape + (1,)))
    # Set each channel of the img array to the value given by the input color
    # tuple. The color must be applied only to pixels location in the img
    # array where the corresponding pixel location in fmask array has value
    # equal to 1. Note that img is a 3d array and fmask is a 2d array. You
    # may do this task in channel-wise manner.

    return img


def color_w_noisy_color(fmask, mean, deviation):
    """
    Colors layer with constant color, then draws random integer uniformly
    from [mean-deviation;mean+deviation] and adds it to the image, generating
    a noisy image.

    Input: fmask -- binary mask indicating the shape that is to be coloured.
    Is 1 at positions which shall be coloured and 0 elsewhere. mean  -- mean
    color, (r,g,b) tuple deviation -- range within which to vary mean color

    Output:
    numpy array of dimensions (x,y,3) representing 3-channel image
    """

    ## Step 3


    # Generate an image with the 'mean' color provided in the function input.
    # You may use the color_w_constant_color from the previous function.
    # img = ...
#     img = np.zeros((fmask.shape + (3,)), dtype = np.int16)
#     img[:, :, 0] = mean[0]
#     img[:, :, 1] = mean[1]
#     img[:, :, 2] = mean[2] 
    img = np.tile(np.array(mean, dtype = np.int16), (fmask.shape + (1,)))

    # Change the data type of img array to 16 bit integer so that it may
    # support negative values and values greater than 255 to avoid overflows



    # Generate a random integer array with values draw uniformly between the
    # range [-deviation, deviation]. The random array must be the same shape
    # as the 'img' variable from the previous line. Use the variable name
    # 'noise' for this array.
    # Refer: https://numpy.org/doc/1.13/reference/generated/numpy.random.randint.html

    noise = np.random.randint(-deviation, deviation+1, size=img.shape)

    # Add the two arrays 'img' and 'noise'. Clip/Threshold the values in the
    # resulting array such that the maximum value of the elements is 255 and
    # minimum value is 0.
    img =  noise + img
    img = np.maximum(img, 0)
    img = np.minimum(img, 255)
    
    # Convert the result to datatype as 8-bit unsigned integer and return the
    # result.
    return np.array(img, dtype=np.uint8)

def color_w_image(fmask, folder, rotate):
    """
    Picks a random color from ([mean[0]-lb;mean[0]+ub],...) and colors
    layer with this color.

    Input:
    fmask -- binary mask indicating the shape that is to be coloured
    folder_name -- name of the folder containing the images


    Output:
    numpy array of dimensions (x,y,3) representing 3-channel image
    """

  # Delete this line before you start implementing.

    ## Step 9 Initialize empty numpy array with datatype 8-bit unsigned
    # integer using variable name 'img'
    img = np.zeros(fmask.shape, dtype = np.uint8)


    # Get the list of images in 'folder' using the get_image_list function

    img_list = get_image_list(folder)

    # Using the random.choice function, randomly select one image from the
    # list and store that image into a variable named 'true_image'.
    true_img_path = random.choice(img_list)
    true_image = imread(true_img_path)

    # If 'rotate' is equal one,then we are required to randomly rotate the
    # image. The rotation is done in steps of 90 degrees.

    if rotate == 1:
#         pass
        # Select a random integer between 0 and 4 and store it in a variable 'step'.
        # step = ...

        step = random.choice([1, 2, 3])


        # Use the numpy rot90 function and rotate the image. Also provide the
        # step variable to this function.
        
        true_image = np.rot90(true_image, step)


    # For all pixel positions in fmask where the value is equal to 1,
    # copy the pixel values from the 'true_image' array
    true_image[fmask == 0, :] = 0
    img = true_image
    return img


# Public API Exporting a registry instead of the functions allows us to
# change the implementation whenever we want. The registry maps that function
# names (of type string) to the actual functions.
COLOR_FCT_REGISTRY = {
    'constant': color_w_constant_color,
    'noisy'   : color_w_noisy_color,
    'image'   : color_w_image
}
