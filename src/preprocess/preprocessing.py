import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import pandas as pd
import numpy as np

abs_path = "C:\\Dev\\Smart Car Project\\auton-car-nnetwork\\data\\"


def convert_raw_data(angles_file, image_dir="frames", output_file="output", steering_offset=-9, save=False):
    angels_df = pd.read_csv(abs_path + angles_file)

    arr = os.listdir(abs_path + image_dir)
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
    final_df['angle'] = (((final_df['angle'] - min) / (max - min)) - 0.5) * 2

    final_df['angle'] = final_df['angle'].round(decimals=3)

    # print(final_df.describe())

    zero_entries = final_df[final_df['angle'] == 0.0].index.tolist()
    print(int(len(zero_entries)*0.8))
    drop_ind = np.random.choice(zero_entries, size=int(len(zero_entries)*0.8))
    final_df = final_df.drop(drop_ind)

    if save:
        final_df.to_csv(abs_path + output_file + "-Mi" + str(min) + "-Ma" + str(max) + "-O" + str(steering_offset) + ".csv", index=False)

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


convert_raw_data("angels.csv", save=True)
