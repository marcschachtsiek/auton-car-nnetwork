import pandas as pd
import os
from keras.preprocessing.image import ImageDataGenerator


abs_path = "C:\\Development\\Smart Car Project\\auton-car-nnetwork\\"

def convert_raw_data(angles_file, image_dir, output_file="output", steering_offset=-9, save=False):
    angels_df = pd.read_csv(abs_path + 'data\\' + angles_file)

    arr = os.listdir(abs_path + 'data\\' + image_dir)
    image_df = read_image_timestamp(arr)

    angle_arr = []

    def match_angle(row):

        hour = int(row['hour'])
        min = int(row['minute'])
        sec = int(row['second'])
        usec = int(row['microsec'])

        match = angels_df[(angels_df['hour'] < hour) |
                          (angels_df['hour'] == hour) & (angels_df['minute'] < min) |
                          (angels_df['hour'] == hour) & (angels_df['minute'] == min) & (angels_df['second'] < sec) |
                          (angels_df['hour'] == hour) & (angels_df['minute'] == min) & (angels_df['second'] == sec) & (angels_df['microsec'] <= usec)]

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
    min = final_df['angle'].min()
    max = final_df['angle'].max()
    final_df['angle'] = (final_df['angle'] - min) / (max - min)

    if save:
        final_df.to_csv(output_file + "-Mi" + str(min) + "-Ma" + str(max) + "-O" + str(steering_offset) + ".csv", index=False)

    return final_df, min, max


def read_image_timestamp(filename_array):
    df = pd.DataFrame(columns=['filename', 'hour', 'minute', 'second', 'microsec'])
    df['filename'] = filename_array

    hour_array = []
    min_array = []
    sec_array = []
    usec_array = []

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


def load_dataset_dataframe(csv_file):
    return pd.read_csv(abs_path + 'src\\preprocess\\' + csv_file, )


def get_dataset_generators_from_dataframe(dataframe, image_dir, x_label, y_label):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = datagen.flow_from_dataframe(dataframe=dataframe, directory=abs_path + "data\\" + image_dir, validate_filenames=False,
                                                  x_col=x_label, y_col=y_label, class_mode="raw", seed=42, target_size=(240, 320),
                                                  subset="training")

    valid_generator = datagen.flow_from_dataframe(dataframe=dataframe, directory=abs_path + "data\\" + image_dir, validate_filenames=False,
                                                  x_col=x_label, y_col=y_label, class_mode="raw", seed=42, target_size=(240, 320),
                                                  subset="validation")

    return train_generator, valid_generator


convert_raw_data("angels.csv", "images", save=True)