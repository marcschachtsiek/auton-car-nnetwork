import glob
import os
import pandas as pd
import cv2


def remove(dataset, dataset_file="dataset-aug.csv"):
    """
    remove([dataset_file])
    .   @brief Removes augmented images from a dataset.
    .
    .   The function remove(), removes all files starting with "m-" from the folder "frames" in the dataset directory.
    .   This removes images created with the mirror_images() function. The function also removes the accompanying CSV
    .   definition file which is by default labelled "dataset-aug.csv"
    .
    .   @param dataset Folder name of dataset.
    .   @param dataset_file Filename of the CSV definition file for the augmented dataset.
    """

    aug_files = glob.glob(dataset + "\\frames\\m-*")
    files = glob.glob(dataset + "\\" + dataset_file)

    for f in aug_files + files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))


def augment(dataset, dataset_def):
    """
    augment(dataset_def) -> dataframe
    .   @brief Augments the dataset.
    .
    .   The function augment(), augments a given dataset by mirroring all images with a steering angle != 0. It copies,
    .   mirrors and saves all the new images in the "frames" folder and creates a new augmented dataset definition
    .   dataframe that is returned.
    .
    .   @param dataset Folder name of dataset.
    .   @param dataset_def Dataset definition dataframe loaded with utilities.csv_load().
    """

    data = []

    def mirror(row):
        angle = float(row["angle"])
        if angle != 0.0:
            filename = row["filename"]
            img = cv2.imread(dataset + "\\frames\\" + filename)
            flip_img = cv2.flip(img, 1)
            cv2.imwrite(dataset + "\\frames\\m-" + filename, flip_img)
            data.append(["m-" + filename, -angle])

    dataset_def.apply(mirror, axis=1)

    data_frame = pd.DataFrame(data, columns=["filename", "angle"])
    return pd.concat([dataset_def, data_frame])
