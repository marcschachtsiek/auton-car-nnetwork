import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Cropping2D, Lambda
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from src.preprocess import preprocessing as prep


def fit_model(train_gen, valid_gen):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    print("Model:")

    model = Sequential()

    model.add(Cropping2D(cropping=((140, 0), (0, 0)), input_shape=(240, 320, 3)))
    model.add(Lambda(lambda x: x/255.0 - 0.5))

    model.add(Conv2D(16, (5, 5), padding='valid', activation='relu'))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (5, 5), padding='valid', activation='relu'))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5), padding='valid', activation='relu'))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Conv2D(64, (5, 5), padding='valid', activation='relu'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))

    model.compile(optimizer=Adam(lr=1.0e-4), loss=MeanSquaredError(), metrics=["accuracy"])

    print(model.summary())

    STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
    STEP_SIZE_VALID = valid_gen.n // valid_gen.batch_size

    history = model.fit(x=train_gen, steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_gen, validation_steps=STEP_SIZE_VALID, epochs=20)

    model.save("model.h5")
    print("Model saved.")

    # Plot history: MSE -- from https://www.machinecurve.com/index.php/2019/10/08/how-to-visualize-the-training-process-in-keras/#visualizing-the-mse
    plt.plot(history.history['val_loss'], label='MSE (validation data)')
    plt.plot(history.history['loss'], label='MSE (training data)')
    plt.title('MSE for Steering Angle Prediction')
    plt.ylabel('MSE value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper right")
    plt.show()


def evaluate_model(filename, valid_gen):
    model = keras.models.load_model(filename)
    y_pred = model.predict(valid_gen)

    plt.plot([a_tuple[0] for a_tuple in y_pred], label='Predicted')
    plt.plot(valid_gen.labels, label='Measured')
    plt.title('Predicted and measured steering angle values')
    plt.ylabel('Normalised Steering Angle Value')
    plt.xlabel('')
    plt.legend(loc="upper right")
    plt.show()


df = prep.load_dataset_dataframe("output-Mi30-Ma150-O-9.csv")
train_gen, valid_gen = prep.get_dataset_generators_from_dataframe(df, "images", "filename", "angle")
fit_model(train_gen, valid_gen)
evaluate_model("model.h5", valid_gen)
