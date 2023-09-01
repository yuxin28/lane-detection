import torch
import numpy as np
import sys

from data.image_dataset import ImageDataset,convert_tensor_to_numpy,convert_numpy_to_tensor

test_register= {
    # name : [type, min, max]
    'image_tensor':[torch.float32,0,1],
    'image_numpy':[np.uint8,0,255],
    'label_tensor':[torch.int64,0,1],
    'label_numpy':[np.uint8,0,255],
}


def test_array_type(input_array, test_type) -> None:
    """Test if array has correct datatype as per the test_register"""

    array_name = test_type
    expected_type = test_register[test_type][0]

    if input_array.dtype == expected_type:
        print('PASS: {} has correct datatype {}'.format(array_name,input_array.dtype))
    else:
        print('FAIL: {} is {}. Expected datatype {}'.format(array_name,input_array.dtype,expected_type))


def test_array_range(image, test_type=None) -> None:
    """Test if the array is within the expected range defined by test_register."""

    array_name = test_type
    max_value = image.max()
    min_value = image.min()
    expected_min_value = test_register[test_type][1]
    expected_max_value = test_register[test_type][2]

    if max_value > expected_max_value or min_value < expected_min_value :
        print('FAIL: {} in range from {} to {}. Expected range {} to {}'.format(array_name, min_value,max_value,expected_min_value,expected_max_value))
    else:
        print('PASS: {} in correct range from {} to {}.'.format(array_name, expected_min_value,expected_max_value))

def test_array_shape(input_shape,image_shape,test_type=None) -> None:
    input_shape = list(input_shape)
    image_shape = list(image_shape)

    if test_type == 'image_tensor':
        expected_shape = [input_shape[2],input_shape[0],input_shape[1]]
    else:
        expected_shape = [input_shape[1],input_shape[2],input_shape[0]]
    print('Input shape:   ',input_shape)
    print('Expected shape:',expected_shape)
    print('Output shape:  ',image_shape)

    if np.array_equal(expected_shape,image_shape):
        print('PASS: correct array shape after conversion')
    else:
        print('FAIL: incorrect array shape mismatch after conversion')


def test_conversion() -> None:
    """Test the numpy <-> tensor conversion functions."""
    print('A. Test converson of numpy array to torch tensor')
    dim = np.random.choice(range(5,10),2,replace=False)
    np_in = np.zeros((dim[0],dim[1],3),dtype=np.uint8)
    np_in[:,:,1] = 127
    np_in[:,:,2] = 255

    tensor_out = convert_numpy_to_tensor(np_in)
    test_array_type(tensor_out,'image_tensor')
    test_array_range(tensor_out,'image_tensor')
    test_array_shape(np_in.shape,tensor_out.shape,'image_tensor')

    print('\nB. Test conversion from tensor in GPU to numpy array')
    tensor_out = tensor_out.to(device="cuda")
    np_out = convert_tensor_to_numpy(tensor_out)
    test_array_type(np_out,'image_numpy')
    test_array_range(np_out,'image_numpy')
    test_array_shape(tensor_out.shape,np_out.shape,'image_numpy')
    if np.array_equal(np_out,np_in):
        print('PASS: Conversion test')
    else:
        print('FAIL: Conversion test')





