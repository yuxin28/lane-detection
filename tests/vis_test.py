import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.io import imread

from data.image_dataset import ImageDataset,convert_tensor_to_numpy,convert_numpy_to_tensor
from pipelines import PROJECT_PATH
from utils.visualization import get_overlay_image

TEST_FIG_PATH = os.path.join(PROJECT_PATH,'tests','figs')
input_image_path = os.path.join(TEST_FIG_PATH,'img1.jpg')
label_image_path = os.path.join(TEST_FIG_PATH,'label1.png')
overlay_image_path = os.path.join(TEST_FIG_PATH,'overlay.png')


def test_overlay(test):
    # name : [type, min, max]
    img = imread(input_image_path)
    label = imread(label_image_path).astype(np.bool)
    overlay_expected = imread(overlay_image_path)
    img_tensor = convert_numpy_to_tensor(img)
    label_tensor = convert_numpy_to_tensor(label)

    


    fig, ax = plt.subplots(2, 2, figsize=(15, 7))
    ax[0,0].set_title("Image")
    ax[0,0].imshow(img)
    ax[0,1].set_title("Label")
    ax[0,1].imshow(label, cmap='gist_gray')
    ax[1,0].set_title("Expected Overlay Image")
    ax[1,0].imshow(overlay_expected)
    if test:
        overlay_tensor = get_overlay_image(img_tensor,label_tensor)
        overlay = convert_tensor_to_numpy(overlay_tensor)
    else:
        overlay = np.zeros_like(img)

    ax[1,1].set_title("Your Implementation")
    ax[1,1].imshow(overlay)
    [i.axis('off') for i in ax.flatten()]
    plt.show()




