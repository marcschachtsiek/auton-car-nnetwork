import pandas as pd
import os


def convert_raw_data(angles_file, images_folder):
    angels_df = pd.read_csv('../data/' + angles_file)

    arr = os.listdir('../data/' + images_folder)
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
    print(final_df)


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


convert_raw_data('angles.csv', 'images')
