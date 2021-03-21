import os

import numpy as np
from sklearn.metrics import mean_squared_error

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import keras
from tensorflow.python.keras.layers import Resizing
from keras import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Cropping2D, Lambda
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
import matplotlib.pyplot as plt

import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


def load_data(csv_file, data_path, image_dir="frames", pre_shuffle=False, shuffle=True, batch_size=32, seed=42,
              target_size=(240, 320)):
    dataframe = pd.read_csv(data_path + csv_file)

    if pre_shuffle:
        dataframe = dataframe.sample(frac=1).reset_index(drop=True)

    datagen = ImageDataGenerator(validation_split=0.2)
    train_generator = datagen.flow_from_dataframe(dataframe=dataframe, directory=data_path + image_dir,
                                                  validate_filenames=False, x_col='filename', y_col='angle',
                                                  class_mode="raw", seed=seed, target_size=target_size,
                                                  subset="training", batch_size=batch_size, shuffle=shuffle)

    valid_generator = datagen.flow_from_dataframe(dataframe=dataframe, directory=data_path + image_dir,
                                                  validate_filenames=False, x_col='filename', y_col='angle',
                                                  class_mode="raw", seed=seed, target_size=target_size,
                                                  subset="validation", batch_size=batch_size, shuffle=shuffle)

    return train_generator, valid_generator


def model_jnet(crop_top, crop_bottom, crop_left, crop_right):
    model = Sequential()
    model.add(Cropping2D(cropping=((crop_top, crop_bottom), (crop_left, crop_right)), input_shape=(240, 320, 3)))
    model.add(Resizing(height=65, width=320))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))

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
    model.add(Lambda(lambda x: x / 255.0 - 0.5))

    model.add(Conv2D(16, (2, 2), activation='relu'))

    model.add(Flatten())
    model.add(Dense(1))

    return model


def model_pilotnet(crop_top, crop_bottom):
    model = Sequential()
    model.add(Cropping2D(cropping=((crop_top, crop_bottom), (0, 0)), input_shape=(240, 320, 3)))
    model.add(Resizing(height=65, width=320))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))

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

    return model


def compile_model(model, lr):
    model.compile(optimizer=Adam(lr=lr), loss=MeanSquaredError(), metrics=['accuracy'])
    print(model.summary())
    return model


def fit_model(model, train_gen, valid_gen, data_path, history_filename='history.csv', epochs=20, max_queue_size=10,
              workers=1, model_format=''):

    checkpoint = ModelCheckpoint(data_path + "\\" + 'model-{epoch:03d}-{val_loss:.3f}' + model_format,
                                 monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

    step_size_train = train_gen.n // train_gen.batch_size
    step_size_valid = valid_gen.n // valid_gen.batch_size

    history = model.fit(x=train_gen, steps_per_epoch=step_size_train, callbacks=[checkpoint],
                        validation_data=valid_gen, validation_steps=step_size_valid, epochs=epochs,
                        max_queue_size=max_queue_size, workers=workers)

    df = pd.DataFrame(history.history)
    df.to_csv(data_path + "\\" + history_filename, index=False)


def plot_feature_maps(folder, model_file, img_filename, ixs, plot_sq, figsizes, data_path):
    model = keras.models.load_model(data_path + folder + model_file)

    # https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
    outputs = [model.layers[i].output for i in ixs]
    model = Model(inputs=model.inputs, outputs=outputs)

    img = load_img(data_path + img_filename)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    feature_maps = model.predict(img)

    for index, fmap in enumerate(feature_maps):

        rows, cols = plot_sq[index][0], plot_sq[index][1]

        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsizes[index],
                                gridspec_kw={'hspace': 0, 'wspace': 0}, squeeze=False)

        ix = 1
        for x in range(rows):
            for y in range(cols):

                for spine in axs[x, y].spines.values():
                    spine.set_edgecolor('blue')

                axs[x, y].set_xticks([])
                axs[x, y].set_yticks([])
                axs[x, y].imshow(fmap[0, :, :, ix - 1], cmap='gray')
                ix += 1
        plt.grid()
        plt.savefig(data_path + folder + "\\figure-layer" + str(ixs[index]) + ".png")
        plt.show()


def load_history(path, filename="history.csv"):
    return pd.read_csv(path + "\\" + filename)


def plot_history(history, out_path, out_filename='history.png', figsize=(5, 3), linewidth=1):

    # Plot history: MSE -- from https://www.machinecurve.com/index.php/2019/10/08/how-to-visualize-the-training
    # -process-in-keras/#visualizing-the-mse
    plt.figure(figsize=figsize)
    plt.plot(history['loss'], label='MSE (training data)', linewidth=linewidth)
    plt.plot(history['val_loss'], label='MSE (validation data)', linewidth=linewidth)
    plt.title('MSE for Steering Angle Prediction')
    plt.ylabel('MSE value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper right", fontsize='small')
    plt.savefig(out_path + "\\" + out_filename, dpi=1200, bbox_inches="tight")
    plt.show()


def load_model(path, filename):
    return keras.models.load_model(path + "\\" + filename)


def load_best_model(path):
    return False


def get_predictions(model, sample_path, csv_file, target_size, percentage=0.01, seed=42, batch_size=64):
    dataframe = pd.read_csv(sample_path + "\\" + csv_file)

    datagen = ImageDataGenerator(validation_split=percentage)
    valid_generator = datagen.flow_from_dataframe(dataframe=dataframe, directory=sample_path + "\\frames",
                                                  validate_filenames=False, x_col='filename', y_col='angle',
                                                  class_mode="raw", seed=seed, target_size=target_size,
                                                  subset="validation", batch_size=batch_size, shuffle=False)

    y_pred = model.predict(valid_generator)

    y_pred_a = [a_tuple[0] for a_tuple in y_pred]
    labels = [label for label in valid_generator.labels]

    return y_pred_a, labels


def plot_predictions(preds_list, plot_labels, labels, out_path, out_filename='predictions.png', figsize=(5, 3),
                     linewidth=1, fontsize='small', title='Predicted vs Ground Truth Steering Angle Values'):

    if type(preds_list[0]) is not list:
        preds_list = [preds_list]

    if type(plot_labels) is not list:
        plot_labels = [plot_labels]

    plt.figure(figsize=figsize)
    plt.plot(labels, label='Measured', linewidth=linewidth, zorder=5)

    for preds, label in zip(preds_list, plot_labels):
        plt.plot(preds, label=label, linewidth=linewidth, zorder=1)

    plt.title(title)
    plt.ylabel('Normalised Steering Angle Value')
    plt.xlabel('Sample')
    plt.legend(loc="upper left", fontsize=fontsize)
    plt.savefig(out_path + "\\" + out_filename, dpi=1200, bbox_inches="tight")
    plt.show()


def plot_mse(preds_list, labels, x_vals, extra_vals, extra_txt, out_path, out_filename='mse_graph.png', figsize=(5, 3),
             linewidth=1, fontsize='small', title='Predicted vs Ground Truth Steering Angle Values'):

    if type(preds_list[0]) is not list:
        preds_list = [preds_list]

    mse = []
    for preds in preds_list:
        mse.append(mean_squared_error(labels, preds))

    plt.figure(figsize=figsize)
    plt.plot(x_vals, mse, label='Mean Square Error', linewidth=linewidth)
    plt.axhline(y=mean_squared_error(labels, extra_vals), label=extra_txt, ls='--', color='r')
    plt.title(title)
    plt.ylabel('MSE')
    plt.xlabel('Blur Strength')
    plt.legend(loc="center left", fontsize=fontsize)
    plt.savefig(out_path + "\\" + out_filename, dpi=1200, bbox_inches="tight")
    plt.show()
