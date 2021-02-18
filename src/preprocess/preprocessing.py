import os
import pandas as pd


def convert_raw_data(angles_file, abs_path, output_file='output', steering_offset=-9, save=False):
    angels_df = pd.read_csv(abs_path + angles_file)

    arr = os.listdir(abs_path + 'frames')
    image_df = read_images(arr)

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
    min_val = final_df['angle'].min()
    max_val = final_df['angle'].max()
    final_df['angle'] = (((final_df['angle'] - min_val) / (max_val - min_val)) - 0.5) * 2

    final_df['angle'] = final_df['angle'].round(decimals=5)

    if save:
        final_df.to_csv(abs_path + output_file + ".csv", index=False)

    return final_df, min_val, max_val


def read_images(filename_array):
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
