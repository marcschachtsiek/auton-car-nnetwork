import os
import glob


def remove(data_path):
    files = glob.glob(data_path + "\\frames\\m-*")

    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))

    files2 = glob.glob(data_path + "\\output-*")

    for f2 in files2:
        try:
            os.remove(f2)
        except OSError as e:
            print("Error: %s : %s" % (f2, e.strerror))
