
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from logger_config import logger

def create_datasets(
    csv_file_path: str, 
    qupath_feature_path: str,
    metadata_file: str,
    image_dir_path: str,
    nuc_feature: list,
    current_path: str,
    target: int, 
    seed: int,
    image_count: int,
    test_size: float,
):
    # Load nucleus features (from QuPath) and set index to 'Image'
    data = pd.read_csv(qupath_feature_path).set_index("Image")

    # Select all columns for use
    first_n_rows_data = data.iloc[:, :]

    # Extract image file identifiers from filenames
    image_file = [i.split("_")[0] for i in os.listdir(image_dir_path)]
    image_filenames_df = pd.DataFrame(image_file, columns=['image_file']).set_index('image_file')

    # Merge features with image file references
    merged_data = pd.concat([image_filenames_df, first_n_rows_data], axis=1, join="inner")
    merged_data['individual ID'] = merged_data.index.str.split('-').str[:2].str.join('-')
    merged_data['image_file'] = merged_data.index

    # Load metadata
    a = pd.read_csv(metadata_file).set_index("SUBJID")
    b = a.copy()

    # One-hot encode categorical metadata
    one_hot_encoded = pd.get_dummies(b['SEX'], prefix='Sex').astype(int)
    one_hot_encoded2 = pd.get_dummies(b['RACE'], prefix='RACE').astype(int)
    one_hot_encoded3 = pd.get_dummies(b['DTHHRDY'], prefix='DTHHRDY').astype(int)

    b = pd.concat([b.drop('SEX', axis=1), one_hot_encoded], axis=1)
    b = pd.concat([b.drop('RACE', axis=1), one_hot_encoded2], axis=1)
    b = pd.concat([b.drop('DTHHRDY', axis=1), one_hot_encoded3], axis=1)

    # Merge metadata with nucleus/image data using individual ID
    merged_data = merged_data.merge(b, how="inner", right_on=b.index, left_on="individual ID")
    merged_data = merged_data.drop_duplicates(subset='individual ID', keep='first')

    # Select final columns for modeling
    merged_data = merged_data[
        ['HGHT', 'WGHT', 'BMI', 'TRDNISCH', 'drinkindex', 'smokeindex', 'TRVNTSR',
         'Sex_1', 'Sex_2', 'RACE_1', 'RACE_2', 'RACE_3', 'RACE_4', 'RACE_98', 'RACE_99',
         'DTHHRDY_0.0', 'DTHHRDY_1.0', 'DTHHRDY_2.0', 'DTHHRDY_3.0', 'DTHHRDY_4.0',
         'image_file', 'AGE'] + nuc_feature
    ]

    feature_input_shape = len(merged_data.columns) - 2  # exclude image_file and AGE

    # Normalize numerical features
    scale_feature = ['HGHT', 'WGHT', 'BMI', 'TRDNISCH', 'drinkindex', 'smokeindex'] + nuc_feature
    for col in scale_feature:
        scaler = StandardScaler()
        merged_data[col] = scaler.fit_transform(merged_data[[col]])

    # Drop rows with any missing values
    merged_data = merged_data.dropna()

    # Shuffle and save data snapshot
    shuffled_data = merged_data.sample(frac=1, random_state=seed)
    shuffled_data.to_csv(os.path.join(current_path, "data_with_genotype.csv"))

    # Extract inputs and outputs
    features = shuffled_data[['image_file']]
    labels = shuffled_data["AGE"]
    meta_nucleus_feature = shuffled_data[
        ['HGHT', 'WGHT', 'BMI', 'TRDNISCH', 'drinkindex', 'smokeindex', 'TRVNTSR',
         'Sex_1', 'Sex_2', 'RACE_1', 'RACE_2', 'RACE_3', 'RACE_4', 'RACE_98', 'RACE_99',
         'DTHHRDY_0.0', 'DTHHRDY_1.0', 'DTHHRDY_2.0', 'DTHHRDY_3.0', 'DTHHRDY_4.0'] + nuc_feature
    ]

    # Split data into train/test/validation sets
    X_temp, X_test, X_nucleus_temp, X_nucleus_test, y_temp, y_test = train_test_split(
        features, meta_nucleus_feature, labels, test_size=test_size, random_state=seed)

    X_train, X_val, X_nucleus_train, X_nucleus_val, y_train, y_val = train_test_split(
        X_temp, X_nucleus_temp, y_temp, test_size=0.2, random_state=seed)

    return X_train, X_nucleus_train, y_train, X_val, X_nucleus_val, y_val, X_test, X_nucleus_test, y_test
