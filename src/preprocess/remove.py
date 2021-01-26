import os
import glob

files = glob.glob("C:/Dev/Smart Car Project/auton-car-nnetwork/data/frames/m-*")

for f in files:
    try:
        os.remove(f)
    except OSError as e:
        print("Error: %s : %s" % (f, e.strerror))

files2 = glob.glob("C:/Dev/Smart Car Project/auton-car-nnetwork/data/output-*")

for f2 in files2:
    try:
        os.remove(f2)
    except OSError as e:
        print("Error: %s : %s" % (f2, e.strerror))
