import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import keras
import tensorflow as tf
from tensorflow.python.keras.layers import Resizing
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Cropping2D, Lambda
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
import matplotlib.pyplot as plt

import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

abs_path = "C:\\Dev\\Smart Car Project\\auton-car-nnetwork\\data\\"


def load_data(csv_file, image_dir="frames", x_label="filename", y_label="angle"):
    dataframe = pd.read_csv(abs_path + csv_file)

    print(dataframe['angle'].nunique())

    dataframe.hist(column=['angle'])
    plt.show()

    datagen = ImageDataGenerator(validation_split=0.2)
    train_generator = datagen.flow_from_dataframe(dataframe=dataframe, directory=abs_path + image_dir,
                                                  validate_filenames=False, x_col=x_label, y_col=y_label, class_mode="raw",
                                                  seed=42, target_size=(240, 320), subset="training", batch_size=32)

    valid_generator = datagen.flow_from_dataframe(dataframe=dataframe, directory=abs_path + image_dir,
                                                  validate_filenames=False, x_col=x_label, y_col=y_label, class_mode="raw",
                                                  seed=42, target_size=(240, 320), subset="validation", batch_size=32)

    return train_generator, valid_generator


def model_jnet():
    model = Sequential()
    model.add(Cropping2D(cropping=((0, 100), (0, 0)), input_shape=(240, 320, 3)))
    model.add(Resizing(height=65, width=320))
    model.add(Lambda(lambda x: x/255.0 - 0.5))

    model.add(Conv2D(16, (5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))

    model.compile(optimizer=Adam(lr=0.001), loss=MeanSquaredError())
    return model


def fit_model(train_gen, valid_gen):

    model = model_jnet()
    print(model.summary())

    STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
    STEP_SIZE_VALID = valid_gen.n // valid_gen.batch_size

    history = model.fit(x=train_gen, steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_gen, validation_steps=STEP_SIZE_VALID, epochs=20)

    model.save(abs_path + "model.h5")
    print("Model saved.")

    return history


def evaluate_model(filename, valid_gen, history):
    model = keras.models.load_model(abs_path + filename)
    y_pred = model.predict(valid_gen)

    # Plot history: MSE -- from https://www.machinecurve.com/index.php/2019/10/08/how-to-visualize-the-training-process-in-keras/#visualizing-the-mse
    plt.plot(history.history['val_loss'], label='MSE (validation data)')
    plt.plot(history.history['loss'], label='MSE (training data)')
    plt.title('MSE for Steering Angle Prediction')
    plt.ylabel('MSE value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper right")
    plt.show()

    plt.plot([a_tuple[0] for a_tuple in y_pred], label='Predicted')
    plt.plot(valid_gen.labels, label='Measured')
    plt.title('Predicted and measured steering angle values')
    plt.ylabel('Normalised Steering Angle Value')
    plt.xlabel('')
    plt.legend(loc="upper right")
    plt.show()


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_gen, valid_gen = load_data("output-Mi30-Ma150-O-9 - aug.csv")
history = fit_model(train_gen, valid_gen)
evaluate_model("model.h5", valid_gen, history)
