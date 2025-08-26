
import os
import random
import numpy as np
import pandas as pd
from augumentation import selection
from logger_config import logger

# Check if a patch is valid based on shape and missing value threshold
def is_valid_patch(arr):
    return arr.shape[0] == 32 and arr.shape[1] == 32 and arr.shape[2] == 128 and np.isnan(arr[:, :, 1]).sum() < 450

# Process and store a valid patch with corresponding labels and metadata
def augment_patch(arr, nucleus_label, y, sum_x, sum_nucleus, sum_y, i, Image_ID, occup, is_train_data):
    if is_valid_patch(arr):
        arr1 = np.nan_to_num(arr)
        sum_x.append(arr1)
        sum_nucleus.append(nucleus_label)
        sum_y.append(y)
        Image_ID.append(i)
        occup.append(np.isnan(arr[:, :, 1]).sum() / (32 * 32))

# Extract patches from a slide and add them to the dataset if valid
def process_slide(i, nucleus_label, y_label, sum_x, sum_nucleus, sum_y, Image_ID, occup, is_train_data):
    a = np.load(i + "_features.npy")                      # Load NIC feature array
    b = np.swapaxes(a, 0, 2)                              # Reorder to (H, W, Channels)
    
    unit_list = []
    for j in range(128):                                  # Clean each channel
        c = b[:, :, j][~np.isnan(b[:, :, j]).all(axis=1)]
        d = (c.T[~np.isnan(c.T).all(axis=1)]).T
        unit_list.append(d)
    
    e = np.array(unit_list)
    f = np.swapaxes(e, 0, 2)                              # Final shape: (H, W, 128)

    f_len, f_width = f.shape[:2]
    f_len_int = f_len // 16
    f_width_int = f_width // 16

    # Extract valid patches by sliding a 32x32 window
    for k in range(f_len_int - 2):
        for p in range(f_width_int - 2):
            patch100 = f[k * 16:(k * 16 + 32), p * 16:(p * 16 + 32), :]
            augment_patch(patch100, nucleus_label, y_label, sum_x, sum_nucleus, sum_y, i, Image_ID, occup, is_train_data)

        # Edge patch (right border)
        patch100 = f[k * 16:(k * 16 + 32), -33:-1, :]
        augment_patch(patch100, nucleus_label, y_label, sum_x, sum_nucleus, sum_y, i, Image_ID, occup, is_train_data)

    # Bottom row patches
    for p in range(f_width_int - 2):
        patch100 = f[-33:-1, p * 16:(p * 16 + 32), :]
        augment_patch(patch100, nucleus_label, y_label, sum_x, sum_nucleus, sum_y, i, Image_ID, occup, is_train_data)

    # Bottom-right corner patch
    patch100 = f[-33:-1, -33:-1, :]
    augment_patch(patch100, nucleus_label, y_label, sum_x, sum_nucleus, sum_y, i, Image_ID, occup, is_train_data)

# Process all slides in the dataset and extract features
def process_all_slides(data, nucleus, labels, is_train_data, image_dir):
    sum_x, sum_nucleus, sum_y, Image_ID, occup = [], [], [], [], []

    os.chdir(image_dir)
    logger.info("Processing all slides: %s", "training" if is_train_data else "testing")

    for index, i in enumerate(data.iterrows()):
        y_label = labels.iloc[index]
        image_file = data.iloc[index]['image_file']
        nucleus_label = nucleus.iloc[index]
        process_slide(image_file, nucleus_label, y_label, sum_x, sum_nucleus, sum_y, Image_ID, occup, is_train_data)

    logger.info("Finished processing.")
    return sum_x, sum_nucleus, sum_y, Image_ID, occup

# Stack and select top-N patches with highest occupancy per WSI
def stack_data(sum_x, sum_nucleus, sum_y, Image_ID, occup):
    image = np.stack(sum_x)
    nucleus = np.stack(sum_nucleus)
    label = np.array(sum_y).reshape(-1)
    occup = np.array(occup)
    Image_ID = np.array(Image_ID)

    image_dic = {i: image[i] for i in range(len(Image_ID))}
    nucleus_dic = {i: nucleus[i] for i in range(len(Image_ID))}

    # Create DataFrame for sorting and grouping
    image_df = pd.DataFrame({
        "Label": label,
        "Image_ID": Image_ID,
        "Occupancy": occup
    })

    # Sort by occupancy within each image ID
    image_df.sort_values(by=["Image_ID", "Occupancy"], ascending=[True, False], inplace=True)

    # Select top 8 patches per image
    top_images = image_df.groupby("Image_ID").head(8)

    # Retrieve top patch tensors
    values_list = [image_dic[key] for key in top_images.index]
    nucleus_list = [nucleus_dic[key] for key in top_images.index]

    x = np.stack(values_list)
    nuc = np.stack(nucleus_list)
    y = np.array(top_images["Label"]).reshape(-1, 1)
    Image_ID_sum = top_images["Image_ID"]
    Occup = np.array(top_images["Occupancy"])

    print(nuc.shape)
    return x, nuc, y, Image_ID_sum, Occup
