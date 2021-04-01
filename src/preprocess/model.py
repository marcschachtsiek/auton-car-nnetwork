import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow as tf

import keras
from keras import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Cropping2D, Lambda
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
import matplotlib.pyplot as plt

import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


def load_data(dataset, dataframe, pre_shuffle=True, shuffle=True, validation_split=0.2, batch_size=32, seed=42,
              target_size=(240, 320)):
    """
    load_data(dataset, dataframe[, pre_shuffle, shuffle, batch_size, seed, target_size])
                -> train_generator, valid_generator
    .   @brief Splits the dataset into a train and validation generator used for training and validation.
    .
    .   The function load_data() converts the dataset into a training generator and a validation generator and returns
    .   those. They are used in training and prediction function.
    .
    .   @param dataset          Folder name of dataset.
    .   @param dataframe        Dataset definition dataframe.
    .   @param pre_shuffle      Boolean, whether pre-shuffling should be performed.
    .   @param shuffle          Boolean, whether the data should be shuffled between epochs.
    .   @param batch_size       Batch_size of the generator
    .   @param seed             Seed value for repeatability
    .   @param target_size      Image target size (height, width).
    .   @param validation_split Percentage of full dataset to be used for validation.
    """

    if pre_shuffle:
        dataframe = dataframe.sample(frac=1).reset_index(drop=True)

    datagen = ImageDataGenerator(validation_split=validation_split)
    train_generator = datagen.flow_from_dataframe(dataframe=dataframe, directory=dataset + "\\frames",
                                                  validate_filenames=False, x_col='filename', y_col='angle',
                                                  class_mode="raw", seed=seed, target_size=target_size,
                                                  subset="training", batch_size=batch_size, shuffle=shuffle)

    valid_generator = datagen.flow_from_dataframe(dataframe=dataframe, directory=dataset + "\\frames",
                                                  validate_filenames=False, x_col='filename', y_col='angle',
                                                  class_mode="raw", seed=seed, target_size=target_size,
                                                  subset="validation", batch_size=batch_size, shuffle=True)

    return train_generator, valid_generator


def model_jnet(crop_top, crop_bottom, crop_left, crop_right, input_shape):
    """
    model_jnet(crop_top, crop_bottom, crop_left, crop_right, input_shape) -> model
    .   @brief Returns a J-Net Keras model with custom cropping and input-shape.
    .
    .   This functions creates a sequential Keras model with the parameters according to the J-Net model by
    .   Kocic et al. (2019). The model is adjusted based off cropping parameters and input shape and returned.
    .
    .   @param crop_top     Amount of pixels cropped from the top of the input image.
    .   @param crop_bottom  Amount of pixels cropped from the bottom of the input image.
    .   @param crop_left    Amount of pixels cropped from the left of the input image.
    .   @param crop_right   Amount of pixels cropped from the right of the input image.
    .   @param input_shape  Tuple input-shape of the image (height, width, depth)
    """

    model = Sequential()

    model.add(Cropping2D(cropping=((crop_top, crop_bottom), (crop_left, crop_right)), input_shape=input_shape))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))

    model.add(Conv2D(16, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))

    return model


def compile_model(model, lr):
    """
    compile_model(model, lr) -> model
    .   @brief Compiles a model with the Adam optimiser and the Mean Squared Error loss function.
    .
    .   This function takes a Keras model and compiles it with the Adam optimiser and the Mean Squared Error
    .   loss function. The supplied learning rate is the starting learning rate for the optimiser
    .
    .   @param model    Keras model to be compiled.
    .   @param lr       Initial learning rate for the Adam optimiser.
    """

    model.compile(optimizer=Adam(lr=lr), loss=MeanSquaredError(), metrics=['accuracy'])
    print(model.summary())
    return model


def fit_model(dataset, model, train_gen, valid_gen, hist_filename='history.csv', epochs=20, max_queue_size=10,
              workers=1, verbose=0):
    """
    fit_model(dataset, model, train_gen, valid_gen[, hist_filename, epochs, max_queue_size, workers, verbose])
    .   @brief
    .
    .   ####
    .
    .   @param dataset
    .   @param model
    .   @param train_gen
    .   @param valid_gen
    .   @param hist_filename
    .   @param epochs
    .   @param max_queue_size
    .   @param workers
    .   @param verbose
    """

    checkpoint = ModelCheckpoint(dataset + "\\" + 'model-{epoch:03d}-{val_loss:.3f}',
                                 monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

    step_size_train = train_gen.n // train_gen.batch_size
    step_size_valid = valid_gen.n // valid_gen.batch_size

    history = model.fit(x=train_gen, steps_per_epoch=step_size_train, callbacks=[checkpoint],
                        validation_data=valid_gen, validation_steps=step_size_valid, epochs=epochs,
                        max_queue_size=max_queue_size, workers=workers, verbose=verbose)

    pd.DataFrame(history.history).to_csv(dataset + "\\" + hist_filename, index=False)


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


def plot_history(dataset, history, out_filename='history.png', figsize=(5, 3), linewidth=1, ylim=None):
    """
    plot_history(dataset, history[, out_filename, figsize, linewidth, ylim])
    .   @brief
    .
    .   ####
    .
    .   @param dataset Folder name of dataset.
    .   @param history
    .   @param out_filename
    .   @param figsize
    .   @param linewidth
    .   @param ylim
    """

    # Plot history: MSE -- from https://www.machinecurve.com/index.php/2019/10/08/how-to-visualize-the-training
    # -process-in-keras/#visualizing-the-mse
    plt.figure(figsize=figsize)
    plt.plot(history['loss'], label='MSE (training data)', linewidth=linewidth)
    plt.plot(history['val_loss'], label='MSE (validation data)', linewidth=linewidth)
    plt.title('MSE for Steering Angle Prediction')
    plt.ylabel('MSE value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper right", fontsize='small')

    if ylim is not None:
        plt.ylim(0, ylim)

    plt.savefig(dataset + "\\" + out_filename, dpi=1200, bbox_inches="tight")
    plt.show()


def load_model(dataset, model_name):
    """
    plot_history(dataset, model_name)
    .   @brief Loads and returns Keras model.
    .
    .   Loads keras model from file and returns the model variable.
    .
    .   @param dataset      Folder name of dataset.
    .   @param model_name   String, name of the model file/folder.
    """

    return keras.models.load_model(dataset + "\\" + model_name)


def load_best_model(path):
    return False


def get_predictions(dataset, model, dataset_def, target_size, percentage=0.01, seed=42, batch_size=64):
    """
    get_predictions(dataset, model, dataset_def, target_size[, percentage, seed, batch_size])
                -> predictions, labels, filenames
    .   @brief
    .
    .   ####
    .
    .   @param dataset      Folder name of dataset.
    .   @param model        Keras model file.
    .   @param dataset_def  Dataset definition dataframe.
    .   @param target_size  Image target size (height, width).
    .   @param percentage   Percentage of full dataset to be predicted [0, 1).
    .   @param seed         Seed value for repeatability.
    .   @param batch_size   Batch_size of the generator.
    """


    datagen = ImageDataGenerator(validation_split=percentage)
    valid_generator = datagen.flow_from_dataframe(dataframe=dataset_def, directory=dataset + "\\frames",
                                                  validate_filenames=False, x_col='filename', y_col='angle',
                                                  class_mode="raw", seed=seed, target_size=target_size,
                                                  subset="validation", batch_size=batch_size, shuffle=False)

    y_pred = model.predict(valid_generator)

    y_pred_a = [a_tuple[0] for a_tuple in y_pred]
    labels = [label for label in valid_generator.labels]
    filenames = [filename for filename in valid_generator.filenames]

    return y_pred_a, labels, filenames


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


def convert_to_tflite(dataset, model_name):
    """
    convert_to_tflite(dataset, model_name)
    .   @brief Loads a model and converts it to a TensorFlow Lite model.
    .
    .   Loads a Keras model from a file and uses the TFLiteConverter to convert it to a TensorFlow Lite .tflite file.
    .
    .   @param dataset      Folder name of dataset.
    .   @param model_name   String, name of the model file/folder.
    """

    model = load_model(dataset, model_name)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open(dataset + "\\" + model_name + '.tflite', 'wb') as f:
        f.write(tflite_model)
