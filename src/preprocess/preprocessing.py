import os
import random

import cv2 as cv
import pandas as pd


def blur_dataset(dataset, blurred_folder, dataset_def, sigma=1, kernel=(0, 0)):
    """
    apply_blur(dataset, blurred_folder, dataset_def[, sigma, kernel]) -> dataframe
    .   @brief
    .
    .   ####
    .
    .   @param dataset
    .   @param blurred_folder
    .   @param dataset_def
    .   @param sigma
    .   @param kernel
    """

    os.makedirs(os.getcwd() + "\\" + blurred_folder + "\\frames")

    def blur(row):
        filename = row["filename"]
        img = cv.imread(dataset + "\\frames\\" + filename)
        img_blur = cv.GaussianBlur(img, kernel, sigma)
        cv.imwrite(blurred_folder + "\\frames\\" + filename, img_blur)

    dataset_def.apply(blur, axis=1)

    return dataset_def.copy()


def mix_datasets(dataset_1, dataset_2, mixed_folder, dataset_def, percentage):
    """
    mixed_dataset(dataset_1, dataset_2, mixed_folder, dataset_def, percentage) -> dataframe
    .   @brief
    .
    .   ####
    .
    .   @param dataset_1
    .   @param dataset_2
    .   @param mixed_folder
    .   @param dataset_def
    .   @param percentage
    """

    os.makedirs(os.getcwd() + "\\" + mixed_folder + "\\frames")

    length = len(dataset_def)
    samples = random.sample(range(0, length), round(percentage * length))

    data = []

    def select(row):
        idx = row.name
        filename = row["filename"]

        if idx in samples:
            img = cv.imread(dataset_1 + "\\frames\\" + filename)
            # prefix = "d1_"
        else:
            img = cv.imread(dataset_2 + "\\frames\\" + filename)
            # prefix = "d2_"

        # cv.imwrite(mixed_folder + "\\frames\\" + prefix + filename, img)
        cv.imwrite(mixed_folder + "\\frames\\" + filename, img)
        # data.append([prefix + filename, row['angle']])
        data.append([filename, row['angle']])

    dataset_def.apply(select, axis=1)

    final_df = pd.DataFrame(data, columns=["filename", "angle"])

    return final_df


def create_dataset(dataset, angles_file, steering_offset=-9, min_angle=30, max_angle=150):
    """
    create_dataset(angles_file[, steering_offset, min_angle, max_angle]) -> dataframe
    .   @brief Creates a dataset definition from raw recorded steering angle and image data with timestamps.
    .
    .   The create_dataset() function generates the dataset definition dataframe and returns it. It reads the original
    .   angles file, performs timestamp matching of recorded frames and steering angles and noramlises the angle data.
    .
    .   @param dataset          Folder name of dataset.
    .   @param angles_file      Filename of the file containing the angles and timestamps.
    .   @param output_file      Output filename for the dataset definition CSV file.
    .   @param steering_offset  Steering offset of the recorded dataset.
    .   @param min_angle        Minimum steering angle, used for normalisation.
    .   @param max_angle        Maximum steering angle, used for normalisation
    """

    angels_df = pd.read_csv(dataset + "\\" + angles_file)
    image_df = _read_images(os.listdir(dataset + "\\frames"))

    angle_arr = []

    def match_angle(row):

        hour = int(row['hour'])
        minu = int(row['minute'])
        sec = int(row['second'])
        usec = int(row['microsec'])

        match = angels_df[(angels_df['hour'] < hour) |
                          (angels_df['hour'] == hour) & (angels_df['minute'] < minu) |
                          (angels_df['hour'] == hour) & (angels_df['minute'] == minu) & (angels_df['second'] < sec) |
                          (angels_df['hour'] == hour) & (angels_df['minute'] == minu) & (angels_df['second'] == sec) &
                          (angels_df['microsec'] <= usec)]

        if not match.empty:
            angle_arr.append(match['angle'].iloc[-1])
        else:
            angle_arr.append(0)

    image_df.apply(match_angle, axis=1)

    final_df = pd.DataFrame(columns=['filename', 'angle'])
    final_df['filename'] = image_df['filename']
    final_df['angle'] = angle_arr

    # Normalisation of data, takes into account steering offset
    final_df['angle'] = final_df['angle'] - steering_offset
    final_df['angle'] = (((final_df['angle'] - min_angle) / (max_angle - min_angle)) - 0.5) * 2

    final_df['angle'] = final_df['angle'].round(decimals=5)

    return final_df


def _read_images(filename_array):
    df = pd.DataFrame(columns=['filename', 'hour', 'minute', 'second', 'microsec'])
    df['filename'] = filename_array

    hour_array, min_array, sec_array, usec_array = [], [], [], []

    def add_columns(filename):
        filename_arr = filename.split("_")
        hour_array.append(filename_arr[3])
        min_array.append(filename_arr[4])
        sec_array.append(filename_arr[5])
        usec_array.append(filename_arr[6])

    df['filename'].apply(add_columns)

    df['hour'] = hour_array
    df['minute'] = min_array
    df['second'] = sec_array
    df['microsec'] = usec_array

    return df
