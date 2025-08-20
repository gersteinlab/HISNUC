import random
import numpy as np
from scipy.ndimage import gaussian_filter
# Rotation
def rot(x_old):
    x_new = np.rot90(x_old,1,axes=(0, 1))
    return x_new

# Flip
def flip(x_old):
    x_new = np.flip(x_old,1)
    return x_new

# Cutting pixels out and adding them into the other side of the image
def shiftwrap(x_old):
    x_new = x_old.copy()
    box = x_old[22:32,:,:]
    x_new[22:32,:,:] = x_new[0:10,:,:]
    x_new[0:10,:,:] = box
    return x_new


def dropout_pixels(image, dropout_rate=0.2):
    """
    Randomly drop out a percentage of pixels in the input image.

    Args:
        image (numpy.ndarray): A 3-dimensional numpy array representing an image
            with dimensions (width, height, depth).
        dropout_rate (float): The percentage of pixels to randomly drop out,
            expressed as a decimal between 0 and 1. Defaults to 0.2.

    Returns:
        A new numpy array with the same dimensions as the input image, but with
        a percentage of pixels randomly set to 0.
    """
    assert image.ndim == 3, "Input must be a 3-dimensional numpy array."
    assert 0 <= dropout_rate <= 1, "Dropout rate must be between 0 and 1."

    # Calculate the number of pixels to drop out.
    num_pixels = int(np.round(dropout_rate * image.shape[0] * image.shape[1]))
    #random.seed(20)
    # Choose random pixel indices to set to 0.
    indices = np.random.choice(image.shape[0] * image.shape[1], num_pixels, replace=False)

    # Set the selected pixels to 0.
    image_flat = image.reshape(-1, image.shape[2])
    image_flat[indices, :] = 0

    # Reshape the modified array to the original shape.
    return image_flat.reshape(image.shape)

def Gaussian_image(image, dropout_rate=0.2, sigma=1):
    """
    Randomly drop out a percentage of pixels in the input image and apply a
    Gaussian filter to the resulting image.

    Args:
        image (numpy.ndarray): A 3-dimensional numpy array representing an image
            with dimensions (width, height, depth).
        dropout_rate (float): The percentage of pixels to randomly drop out,
            expressed as a decimal between 0 and 1. Defaults to 0.2.
        sigma (float): The standard deviation of the Gaussian filter. Larger values
            result in more blurring. Defaults to 1.

    Returns:
        A new numpy array with the same dimensions as the input image, but with
        a percentage of pixels randomly set to 0 and a Gaussian filter applied.
    """
    assert image.ndim == 3, "Input must be a 3-dimensional numpy array."
    assert 0 <= dropout_rate <= 1, "Dropout rate must be between 0 and 1."

    # Convert the input image to float32.
    image = image.astype(np.float32)

    # Apply dropout to the image.
    dropout_image = dropout_pixels(image, dropout_rate)

    # Apply a Gaussian filter to the image.
    filtered_image = gaussian_filter(dropout_image, sigma=sigma)

    return filtered_image


# Random augmentation
def selection(x_old,num):

    if num ==1:
        return Gaussian_image(rot(x_old), dropout_rate=0.2, sigma=0.5)
    if num ==2:
        return Gaussian_image(flip(x_old), dropout_rate=0.2, sigma=0.5)
    if num ==3:
        return flip(rot(x_old))
    if num ==4:
        return Gaussian_image(shiftwrap(x_old), dropout_rate=0.2, sigma=0.5)
    if num == 5:
        return rot(shiftwrap(x_old))
    if num == 6:
        return flip(shiftwrap(x_old))
    if num == 7:
        return rot(flip(shiftwrap(x_old)))
    if num == 8:
        return Gaussian_image(rot(flip(shiftwrap(x_old))),sigma=0.5,dropout_rate=0.2)

        
