import os
from tqdm import tqdm
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from pathlib import Path

def count_images(dataset=''):
    """
    Counts the total number of images in a dataset directory.

    Args:
        dataset (str): Path to the dataset directory.

    Returns:
        int: Total number of images in the dataset.
    """
    img_count = 0
    for _, _, files in os.walk(dataset):
        img_count += len(files)
    return img_count

def get_categories(dataset=''):
    """
    Gets the list of categories (subdirectories) in a dataset directory.

    Args:
        dataset (str): Path to the dataset directory.

    Returns:
        list: List of category names.
    """
    categories = os.listdir(dataset)
    return categories

def rename_images(dataset='', categories=[], filename="rename"):
    """
    Renames images in each category directory of a dataset.

    Args:
        dataset (str): Path to the dataset directory.
        categories (list): List of category names.
        filename (str): New filename prefix.

    Returns:
        None
    """
    for category in categories:
        dataset_name = os.path.join(dataset, category)
        files = os.listdir(dataset_name)
        files.sort()
        with tqdm(total=len(files), desc=f'Renaming images in {category} category') as pbar:
            for i, file in enumerate(files):
                new_name = f"{filename}_{i + 1}.jpg"
                os.rename(os.path.join(dataset_name, file), os.path.join(dataset_name, new_name))
                pbar.update(1)

def load_data_labels(categories=[], dataset=''):
    """
    Loads images and their corresponding labels from a dataset.

    Args:
        categories (list): List of category names.
        dataset (str): Path to the dataset directory.

    Returns:
        tuple: A tuple containing numpy array of images and a list of labels.
    """
    data = []
    labels = []
    img_count = count_images(dataset=dataset)
    with tqdm(total=img_count, desc='Data Labels :- ') as pbar:
        for category in categories:
            path = dataset + '/' + category
            for img in os.listdir(path):
                img_path = os.path.join(path, img)
                image = load_img(img_path, target_size=(224, 224))
                image = img_to_array(image)
                image = preprocess_input(image)
                data.append(image)
                labels.append(category)
                pbar.update(1)
    return np.array(data), labels

def encode_labels(data=[], labels=[]):
    """
    Encodes categorical labels into numerical format.

    Args:
        data (array): Array of image data.
        labels (list): List of categorical labels.

    Returns:
        tuple: A tuple containing numpy array of encoded labels and image data.
    """
    lb = LabelEncoder()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)
    labels = np.array(labels)
    data = np.array(data , dtype="float32")
    return data, labels


def convert_path_to_df(dataset, categories=[]):
    """
    Convert image files in the dataset directory to a DataFrame.

    Args:
        dataset (str): Path to the dataset directory.

    Returns:
        pd.DataFrame: DataFrame containing filepaths and corresponding labels.
    """
    # Convert dataset path to a Path object
    image_dir = Path(dataset)

    # Get filepaths for JPG and PNG files recursively
    file_extensions = ['JPG', 'jpg', 'png', 'PNG']
    filepaths = []
    for category in categories:
        for ext in file_extensions:
            filepaths.extend(list(image_dir.glob(f'{category}/*.{ext}')))

    # Extract labels from directory structure
    labels = [os.path.split(os.path.split(x)[0])[1] for x in filepaths]

    # Create pandas Series for filepaths and labels
    filepaths_series = pd.Series(filepaths, name='Filepath').astype(str)
    labels_series = pd.Series(labels, name='Label')

    # Concatenate filepaths and labels into a DataFrame
    image_df = pd.concat([filepaths_series, labels_series], axis=1)
    
    return image_df

def data_augmentation(train_df, test_df):
    aug = ImageDataGenerator(
		rotation_range=20,
		zoom_range=0.2,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		vertical_flip = True,
		fill_mode="nearest")
	
	# Split the data into three categories.
    train_images = aug.flow_from_dataframe(
		dataframe=train_df,
		x_col='Filepath',
		y_col='Label',
		target_size=(224, 224),
		color_mode='rgb',
		class_mode='categorical',
		batch_size=8,
		shuffle=True,
		seed=42,
		subset='training'
	)

    val_images = aug.flow_from_dataframe(
		dataframe=train_df,
		x_col='Filepath',
		y_col='Label',
		target_size=(224, 224),
		color_mode='rgb',
		class_mode='categorical',
		batch_size=8,
		shuffle=True,
		seed=42,
		subset='validation'
	)

    test_images = aug.flow_from_dataframe(
		dataframe=test_df,
		x_col='Filepath',
		y_col='Label',
		target_size=(224, 224),
		color_mode='rgb',
		class_mode='categorical',
		batch_size=8,
		shuffle=False
	)

    return train_images, val_images, test_images