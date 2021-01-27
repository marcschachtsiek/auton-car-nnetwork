import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import keras
from tensorflow.python.keras.layers import Resizing
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Cropping2D, Lambda
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
import matplotlib.pyplot as plt

import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

data_path = "C:\\Dev\\Smart Car Project\\auton-car-nnetwork\\data\\"


def load_data(csv_file, image_dir="frames"):
    dataframe = pd.read_csv(data_path + csv_file)

    print(dataframe['angle'].nunique())

    datagen = ImageDataGenerator(validation_split=0.2)
    train_generator = datagen.flow_from_dataframe(dataframe=dataframe, directory=data_path + image_dir,
                                                  validate_filenames=False, x_col='filename', y_col='angle', class_mode="raw",
                                                  seed=42, target_size=(240, 320), subset="training", batch_size=32)

    valid_generator = datagen.flow_from_dataframe(dataframe=dataframe, directory=data_path + image_dir,
                                                  validate_filenames=False, x_col='filename', y_col='angle', class_mode="raw",
                                                  seed=42, target_size=(240, 320), subset="validation", batch_size=32)

    return train_generator, valid_generator


def model_jnet(crop_top, crop_bottom):
    model = Sequential()
    model.add(Cropping2D(cropping=((crop_top, crop_bottom), (0, 0)), input_shape=(240, 320, 3)))
    model.add(Resizing(height=65, width=320))
    model.add(Lambda(lambda x: x/255.0 - 0.5))

    model.add(Conv2D(16, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))

    return model


def model_shallow(crop_top, crop_bottom):
    model = Sequential()
    model.add(Cropping2D(cropping=((crop_top, crop_bottom), (0, 0)), input_shape=(240, 320, 3)))
    model.add(Resizing(height=65, width=320))
    model.add(Lambda(lambda x: x/255.0 - 0.5))

    model.add(Conv2D(16, (2, 2), activation='relu'))

    model.add(Flatten())
    model.add(Dense(1))

    return model


def model_alexnet(crop_top, crop_bottom):
    model = Sequential()
    model.add(Cropping2D(cropping=((crop_top, crop_bottom), (0, 0)), input_shape=(240, 320, 3)))
    model.add(Resizing(height=65, width=320))
    model.add(Lambda(lambda x: x/255.0 - 0.5))

    model.add(Conv2D(96, (11, 11), activation='relu'))
    model.add(Conv2D(256, (5, 5), activation='relu'))
    model.add(Conv2D(384, (3, 3), activation='relu'))
    model.add(Conv2D(384, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(4096, activation="relu"))
    model.add(Dense(1000, activation="relu"))
    model.add(Dense(1))
    return model


def model_pilotnet(crop_top, crop_bottom):

    model = Sequential()
    model.add(Cropping2D(cropping=((crop_top, crop_bottom), (0, 0)), input_shape=(240, 320, 3)))
    model.add(Resizing(height=65, width=320))
    model.add(Lambda(lambda x: x/255.0 - 0.5))

    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (3, 3), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model


def fit_model(train_gen, valid_gen, model, folder, history_filename='history.csv', epochs=20):

    model.compile(optimizer=Adam(lr=0.001), loss=MeanSquaredError(), metrics=['accuracy'])
    print(model.summary())

    checkpoint = ModelCheckpoint(data_path + folder + 'model-{epoch:03d}-{val_loss:.3f}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')

    step_size_train = train_gen.n // train_gen.batch_size
    step_size_valid = valid_gen.n // valid_gen.batch_size

    history = model.fit(x=train_gen, steps_per_epoch=step_size_train, callbacks=[checkpoint],
                        validation_data=valid_gen, validation_steps=step_size_valid, epochs=epochs)

    df = pd.DataFrame(history.history)
    df.to_csv(data_path + folder + history_filename, index=False)


def plot_training(filename, hist=None):
    if not hist:
        hist = pd.read_csv(data_path + filename)

    # Plot history: MSE -- from https://www.machinecurve.com/index.php/2019/10/08/how-to-visualize-the-training
    # -process-in-keras/#visualizing-the-mse
    plt.plot(hist['val_loss'], label='MSE (validation data)')
    plt.plot(hist['loss'], label='MSE (training data)')
    plt.title('MSE for Steering Angle Prediction')
    plt.ylabel('MSE value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper right")
    plt.show()


def evaluate_model(filename, valid_generator):
    model = keras.models.load_model(data_path + filename)
    y_pred = model.predict(valid_generator)

    plt.plot([a_tuple[0] for a_tuple in y_pred], label='Predicted')
    plt.plot(valid_generator.labels, label='Measured')
    plt.title('Predicted and measured steering angle values')
    plt.ylabel('Normalised Steering Angle Value')
    plt.xlabel('')
    plt.legend(loc="upper right")
    plt.show()
