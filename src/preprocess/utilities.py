import os
import pandas as pd


def dataset_load(dataset, filename="dataset-aug.csv"):
    """
    csv_load(filename) -> dataframe
    .   @brief Loads a dataset definition file.
    .
    .   Wrapper around read_csv() function from pandas to load a dataset definition file into a dataframe that is
    .   then returned.
    .
    .   @param dataset  Folder name of dataset.
    .   @param filename Filename of the CSV file to load.
    """

    return pd.read_csv(dataset + "\\" + filename)


def dataset_save(dataset, dataset_def, filename="dataset-aug.csv"):
    """
    csv_save(dataset, dataset_def[, filename])
    .   @brief Saves a dataset to a dataset definition file.
    .
    .   Wrapper around to_csv() function from pandas to save a dataset dataframe into a dataset definition CSV file.
    .   The dataframe is saved without indices.
    .
    .   @param dataset      Folder name of dataset.
    .   @param dataset_def  Dataset dataframe to save.
    .   @param filename     Filename of the CSV file to load.
    """

    dataset_def.to_csv(dataset + "\\" + filename, index=False)
