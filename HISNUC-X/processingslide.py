
import os
import numpy as np
import pandas as pd
from augumentation import selection  # Assumes patch selection logic is defined here
from logger_config import logger


def is_valid_patch(patch):
    """
    Check if patch has expected shape and not too many NaNs in the 2nd channel.

    Returns:
    - bool: True if patch is valid
    """
    return patch.shape == (32, 32, 128) and np.isnan(patch[:, :, 1]).sum() < 450


def augment_patch(patch, nucleus_label, gene_expression, patches, nucleus_labels, gene_expressions, image_id, image_ids, occupancy, is_train_data):
    """
    Normalize and append valid patches and labels to their respective lists.
    """
    if is_valid_patch(patch):
        patch_normalized = np.nan_to_num(patch)
        patches.append(patch_normalized)
        nucleus_labels.append(nucleus_label)
        gene_expressions.append(gene_expression)
        image_ids.append(image_id)
        occupancy.append(np.isnan(patch[:, :, 1]).sum() / (32 * 32))


def process_slide(image_id, nucleus_label, gene_expression, patches, nucleus_labels, gene_expressions, image_ids, occupancy, is_train_data):
    """
    Extract overlapping 32x32 patches from feature array and append valid ones.
    """
    features = np.load(image_id + "_features.npy")
    features_transposed = np.swapaxes(features, 0, 2)

    # Remove fully NaN rows/columns from each channel
    valid_features = []
    for channel in range(128):
        ch_data = features_transposed[:, :, channel]
        ch_valid_rows = ch_data[~np.isnan(ch_data).all(axis=1)]
        ch_valid = ch_valid_rows.T[~np.isnan(ch_valid_rows.T).all(axis=1)].T
        valid_features.append(ch_valid)

    valid_features_array = np.array(valid_features)
    valid_features_transposed = np.swapaxes(valid_features_array, 0, 2)

    h, w = valid_features_transposed.shape[:2]
    h_steps = h // 16
    w_steps = w // 16

    # Sliding window patch extraction
    for i in range(h_steps - 2):
        for j in range(w_steps - 2):
            patch = valid_features_transposed[i*16:i*16+32, j*16:j*16+32, :]
            augment_patch(patch, nucleus_label, gene_expression, patches, nucleus_labels, gene_expressions, image_id, image_ids, occupancy, is_train_data)
        # Right edge patch
        patch = valid_features_transposed[i*16:i*16+32, -33:-1, :]
        augment_patch(patch, nucleus_label, gene_expression, patches, nucleus_labels, gene_expressions, image_id, image_ids, occupancy, is_train_data)

    # Bottom edge patches
    for j in range(w_steps - 2):
        patch = valid_features_transposed[-33:-1, j*16:j*16+32, :]
        augment_patch(patch, nucleus_label, gene_expression, patches, nucleus_labels, gene_expressions, image_id, image_ids, occupancy, is_train_data)

    # Bottom-right corner
    patch = valid_features_transposed[-33:-1, -33:-1, :]
    augment_patch(patch, nucleus_label, gene_expression, patches, nucleus_labels, gene_expressions, image_id, image_ids, occupancy, is_train_data)


def process_all_slides(data, nucleus_data, gene_expression_data, is_train_data, IMG_DIR):
    """
    Loop through all image IDs and extract patches.

    Parameters:
    - data: DataFrame with image_file column.
    - nucleus_data: DataFrame with nucleus features (indexed by sample ID).
    - gene_expression_data: DataFrame with gene expression (indexed by sample ID).
    - is_train_data: Whether this is training data (used for logging).
    - IMG_DIR: Directory where image feature .npy files are stored.

    Returns:
    - Tuple of lists for patches, labels, IDs, and occupancy values.
    """
    patches, nucleus_labels, gene_expressions, image_ids, occupancy = [], [], [], [], []
    os.chdir(IMG_DIR)

    logger.info(f"Processing all slides: {'training' if is_train_data else 'testing'}")
    for _, row in data.iterrows():
        image_file = row['image_file']
        nucleus_label = nucleus_data.loc[row.name]
        gene_expression = gene_expression_data.loc[row.name]
        process_slide(image_file, nucleus_label, gene_expression, patches, nucleus_labels, gene_expressions, image_ids, occupancy, is_train_data)

    logger.info("Finished processing all slides.")
    return patches, nucleus_labels, gene_expressions, image_ids, occupancy


def stack_data(patches, nucleus_labels, gene_expressions, image_ids, occupancy):
    """
    Stack raw patch data and select top 8 patches per image based on occupancy.

    Returns:
    - stacked_images: (N, 32, 32, 128)
    - stacked_nucleus: (N, F)
    - selected_gene_expressions: (N, G)
    - selected_image_ids: (N,)
    - selected_occupancy: (N,)
    """
    logger.info("Stacking data...")

    # Convert to arrays
    images = np.stack(patches)
    nucleus_data = np.stack(nucleus_labels)
    gene_expression_data = np.array(gene_expressions)
    occupancy_data = np.array(occupancy)
    image_ids_data = np.array(image_ids)

    # Track shapes
    logger.info(f"Images shape: {images.shape}")
    logger.info(f"Nucleus data shape: {nucleus_data.shape}")
    logger.info(f"Gene expression shape: {gene_expression_data.shape}")
    logger.info(f"Occupancy shape: {occupancy_data.shape}")
    logger.info(f"Image IDs shape: {image_ids_data.shape}")

    # Create mappings for selection
    image_dict = {i: images[i] for i in range(len(image_ids_data))}
    nucleus_dict = {i: nucleus_data[i] for i in range(len(image_ids_data))}
    
    # Combine info into DataFrame
    image_df = pd.DataFrame({
        "Image_ID": image_ids_data,
        "Occupancy": occupancy_data
    })
    gene_expr_df = pd.DataFrame(gene_expression_data, columns=list(range(2, 102)))
    image_df = pd.concat([image_df, gene_expr_df], axis=1)

    # Sort and select top 8 per image
    image_df.sort_values(by=["Image_ID", "Occupancy"], ascending=[True, False], inplace=True)
    top_images = image_df.groupby("Image_ID").head(8)

    logger.info(f"Top images shape: {top_images.shape}")

    selected_images = [image_dict[i] for i in top_images.index]
    selected_nucleus = [nucleus_dict[i] for i in top_images.index]
    stacked_images = np.stack(selected_images)
    stacked_nucleus = np.stack(selected_nucleus)

    logger.info(f"Stacked images shape: {stacked_images.shape}")
    logger.info(f"Stacked nucleus shape: {stacked_nucleus.shape}")

    selected_gene_expressions = top_images.iloc[:, 2:102].values
    selected_image_ids = top_images["Image_ID"]
    selected_occupancy = top_images["Occupancy"].values

    logger.info(f"Selected gene expressions shape: {selected_gene_expressions.shape}")
    logger.info(f"Selected image IDs shape: {selected_image_ids.shape}")
    logger.info(f"Selected occupancy shape: {selected_occupancy.shape}")

    return stacked_images, stacked_nucleus, selected_gene_expressions, selected_image_ids, selected_occupancy
