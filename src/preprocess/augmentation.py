import pandas as pd
import cv2

data_path = "C:\\Dev\\Smart Car Project\\auton-car-nnetwork\\data\\"


def mirror_images(csv_file, out_filename):
    dataframe = pd.read_csv(data_path + csv_file)
    data = []

    def mirror(row):
        angle = float(row["angle"])
        if angle != 0.0:
            filename = row["filename"]
            img = cv2.imread(data_path + "frames\\" + filename)
            flip_img = cv2.flip(img, 1)
            cv2.imwrite(data_path + "frames\\m-" + filename, flip_img)
            data.append(["m-" + filename, -angle])

    dataframe.apply(mirror, axis=1)

    data_frame = pd.DataFrame(data, columns=["filename", "angle"])
    result = pd.concat([dataframe, data_frame])

    result.to_csv(data_path + out_filename, index=False)


mirror_images("output-Mi30-Ma150-O-9.csv", "output-Mi30-Ma150-O-9 - aug.csv")