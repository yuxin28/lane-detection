import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter

from utils.experiment import read_config

DEFAULT_PARAMS = read_config('./config.json')


################################################################################
# HELPER FUNCTIONS : PIL <-> NUMPY CONVERSION

def convert_to_PIL(image, label):
    image = Image.fromarray(image, mode='RGB')
    # mode L => 8 bit unsigned int (grayscale) images
    label = Image.fromarray(label, mode='L')

    return image, label


def convert_to_numpy(image, label):
    image = np.array(image)
    label = np.array(label, dtype=np.uint8)

    return image, label


################################################################################

class VerticalFlip(object):
    """Mirror(Vertical flip)  both image and label"""

    def __init__(self, probability=DEFAULT_PARAMS['vertical_flip_prob']):
        self.probability = probability

    def __call__(self, input_imgs):
        image, label = input_imgs
        # Limit the number of augmented images in dataset by applying augmentation
        # only if random number generated is less than self.probability
        if random.random() < self.probability:
            image = np.flip(image, 1).copy()  # Flip image
            label = np.flip(label, 1).copy()  # Flip label
        return image, label


class GaussianBlur(object):
    """
    Apply blur on the input image using a gaussian filter. The label is not modified.
    """

    def __init__(self, probability=DEFAULT_PARAMS['blur_prob']):
        self.probability = probability

    def __call__(self, input_imgs):
        image, label = input_imgs
        # Limit the number of augmented images in dataset by applying augmentation
        # only if random number generated is less than self.probability
        if random.random() < self.probability:
            radius = random.uniform(0.25, 3)
            image, label = convert_to_PIL(image, label)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

            image, label = convert_to_numpy(image, label)
        return image, label


class Rotate(object):
    """
    Rotate both image and label rotation angle by randomly sampling an angle
     from a uniform distribution with interval [-max_rotation, +max_rotation]
    """

    def __init__(self, probability=DEFAULT_PARAMS['rotate_prob']):
        self.probability = probability

    def __call__(self, input_imgs):
        image, label = input_imgs

        max_rotation = DEFAULT_PARAMS['max_rot']
        # Augmentation is only performed if random number generated is less than
        # self.probability. This limits the number of augmented images in the
        # dataset.
        if random.random() < self.probability:
            ## Step 2a: Remove the pass statement below and complete the missing code.
            
            # Obtain angle using suitable random function
            angel = np.random.uniform(-max_rotation, max_rotation, 1)

            # Convert to PIL
            image = Image.fromarray(image)
            label = Image.fromarray(label)

            # Rotate both image and label using suitable PIL function.
            image = image.rotate(angel)
            label = label.rotate(angel)
            #Convert back to numpy


        return np.array(image), np.array(label)


class GaussianNoise(object):
    """Add Gaussian noise to image only - Gaussian noise is added pixel- and
    channelwise to image - value added added to each channel of each pixel is
    drawn from a normal distribution of mean DEFAULT_PARAMS['mean'] and
    standard deviation DEFAULT_PARAMS['std'] """

    def __init__(self, probability=DEFAULT_PARAMS['gaussian_prob']):
        self.probability = probability

    def __call__(self, input_imgs):
        image, label = input_imgs

        mean = DEFAULT_PARAMS['mean']
        std = DEFAULT_PARAMS['std']
        # Augmentation is only performed if random number generated is less than
        # self.probability. This limits the number of augmented images in the
        # dataset.
        if random.random() < self.probability:
            # Step 2b: Remove the pass statement below and complete the missing code.
            image = image.astype(float)
            noise = np.random.normal(mean, std, image.shape)
            image = np.clip(0, 255, (image + noise)).astype('uint8')
            
            # Convert the image to datatype float

            # Create a noise array using random functions


            # Add the noise array to the image.


            # Ensure that the pixel values are between 0 and 255. If not, clip appropriately.


            # Convert the image back to 8 bit unsigned integer

        return image, label


class ColRec(object):
    """Add colored rectangles of height params['y_size'] and width
       params['x_size'] to image only
        - number  of rectangles is specified by params['num_rec']
        - position is drawn randomly from a uniform distribution
        - value of each color channel is drawn randomly from a uniform
          distribution"""

    def __init__(self, probability=DEFAULT_PARAMS['colrec_prob']):
        self.probability = probability

    def __call__(self, input_imgs):
        image, label = input_imgs

        n_rectangle = random.randint(1, DEFAULT_PARAMS['num_rec'])
        y_size = DEFAULT_PARAMS['y_size']
        x_size = DEFAULT_PARAMS['x_size']

        # Augmentation is only performed if random number generated is less than
        # self.probability. This limits the number of augmented images in the
        # dataset.
        if random.random() < self.probability:
            for i in range(n_rectangle):
                # Step 2c: Remove the pass statement below and complete the missing code.
                x = random.randint(0, 255-x_size)
                y = random.randint(0, 255-y_size)


                # Select a random (y,x) pixel position for top left corner of the rectangle.
                # Since image size is 256 x 256, select point such that
                # x is a random value between 0 and 255-x_size
                # y is a random value between 0 and 255-y_size
                # This is done so that the rectangle does not go outside the image.
                # Note: pixel values must be integers!

                image[y: y+y_size, x: x+x_size] = np.random.choice(range(256), size=3)
                

                # The rectangle will be in the location [y : y+y_size, x : x+x_size]
                # In the input image array, assign a random color to all the pixels inside this
                # rectangle.


                label[y: y+y_size, x: x+x_size] = 0

                # Similarly, for the label array, assign pixel value = 0 for
                # all pixels inside this rectangle





        return image, label


class ZoomIn(object):
    """ - from the original image and label crop a squared box
        - height and width of the box is uniformly drawn from
          [255 * DEFAULT_PARAMS['box_size_min'], 255 * DEFAULT_PARAMS['box_size_max'])
        - position of the box is drawn randomly from a uniform distribution
        - cropped is resized to PIL image of size 256x256"""

    def __init__(self, probability=DEFAULT_PARAMS['zoomin_prob']):
        self.probability = probability

    def __call__(self, input_imgs):
        image, label = input_imgs

        # Apply augmentation only if random number generated is less than
        # probability specified in params

        box_size_min = DEFAULT_PARAMS['box_size_min']
        box_size_max = DEFAULT_PARAMS['box_size_max']
        box_size = np.int_(random.uniform(box_size_min, box_size_max) * 255)

        # Augmentation is only performed if random number generated is less than
        # self.probability. This limits the number of augmented images in the
        # dataset.
        if random.random() < self.probability:
           # Step 2d: Remove the pass statement below and complete the missing code.
           

           # Similar to previous step, select a random (y,x) pixel position for
           # top left corner of the box.
           # Since image size is 256 x 256, select point such that
           # x is a random value between 0 and 255-box_size
           # y is a random value between 0 and 255-box_size
            x = random.randint(0, 255-box_size)
            y = random.randint(0, 255-box_size)



           # Now we know the location of the box.
           # Define a list  such that [top_left_y, top_left_x, bottom_left_y, bottom_left_x]

           # box = ...
            box = [y, x, y+box_size, x+box_size]


            image = image[box[0]: box[2], box[1]: box[3]]
            label = label[box[0]: box[2], box[1]: box[3]]
            image = np.array(Image.fromarray(image).resize((256, 256)))
            label = np.array(Image.fromarray(label).resize((256, 256)))
           # Crop both the image and label using box list and resize it back to
           # (height = 256,width =256)
           # Hint: Refer PIL library for suitable functions. Remember to convert
           # the result back to numpy array before returning the result.


        return image, label


class ZoomOut(object):
    """ A larger black image is created based on the zoom constraints. The image
     is placed in this image at random position. This new image is resize to
     256 x 256."""

    def __init__(self, probability=DEFAULT_PARAMS['zoomout_prob']):
        self.probability = probability

    def __call__(self, input_imgs):
        image, label = input_imgs

        zoom_min = DEFAULT_PARAMS['zoomfac_min']
        zoom_max = DEFAULT_PARAMS['zoomfac_max']
        zoomed_size = np.int_(random.uniform(zoom_min, zoom_max) * 255)
        # Augmentation is only performed if random number generated is less than
        # self.probability. This limits the number of augmented images in the
        # dataset.
        if random.random() < self.probability:
            # Step 2e: Remove the pass statement below and complete the missing code.





            # Create a large 3D black image (zero valued array) of
            # size zoomed_size x zoomed_size
            # black_image_3d = ...

            black_image_3d = np.zeros((zoomed_size, zoomed_size, 3))

            # For the label, we use a 2D black image of same spatial dimension.
            # black_image_2d =
            black_image_2d = np.zeros((zoomed_size, zoomed_size))



            # Similar to previous step, select a random (y,x) pixel position for
            # top left corner of the image/label inside the black image.
            # Since inpput image size is 256 x 256, select point such that
            # x is a random value between 0 and zoomed_size-255
            # y is a random value between 0 and zoomed_size-255
            x = random.randint(0, zoomed_size-256)
            y = random.randint(0, zoomed_size-256)
            black_image_3d[y:y+256, x:x+256] = image
            black_image_2d[y:y+256, x:x+256] = label

            # Now we know the location of the image. Replace the pixel values
            # at this location in the 3d black image with the input image and
            # 2d black image with label respectively.
            image = np.array(Image.fromarray(black_image_3d.astype('uint8')).resize((256, 256)))
            label = np.array(Image.fromarray(black_image_2d.astype('uint8')).resize((256, 256)))

            # Use PIL library to resize the image and label to 256 x 256 as before.


        return image, label
