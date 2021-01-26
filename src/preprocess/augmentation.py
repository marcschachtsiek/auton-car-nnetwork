import pandas as pd
import cv2

import matplotlib.pyplot as plt

abs_path = "C:\\Dev\\Smart Car Project\\auton-car-nnetwork\\data\\"


def mirror_images(csv_file, out_filename):
    dataframe = pd.read_csv(abs_path + csv_file)
    data = []

    def mirror(row):
        angle = float(row["angle"])
        if angle != 0.0:
            filename = row["filename"]
            img = cv2.imread(abs_path + "frames\\" + filename)
            flip_img = cv2.flip(img, 1)
            cv2.imwrite(abs_path + "frames\\m-" + filename, flip_img)
            data.append(["m-" + filename, -angle])

    dataframe.apply(mirror, axis=1)

    data_frame = pd.DataFrame(data, columns=["filename", "angle"])
    # dataframe.append(data_frame, ignore_index=True)
    result = pd.concat([dataframe, data_frame])

    result.to_csv(abs_path + out_filename, index=False)


mirror_images("output-Mi30-Ma150-O-9.csv", "output-Mi30-Ma150-O-9 - aug.csv")