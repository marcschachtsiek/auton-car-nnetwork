from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense, Lambda
from keras.losses import MeanSquaredError
import matplotlib.pyplot as plt
from src.preprocess import preprocessing as prep


def fit_model(train_gen, valid_gen):
    model = Sequential()
    model.add(Conv2D(16, (5, 5), padding='valid', input_shape=(240, 320, 3), activation='relu'))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (5, 5), padding='valid', activation='relu'))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5), padding='valid', activation='relu'))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss=MeanSquaredError(), metrics=["mse"])

    print(model.summary())

    STEP_SIZE_TRAIN = train_gen.n//train_gen.batch_size
    STEP_SIZE_VALID = valid_gen.n//valid_gen.batch_size

    history = model.fit(x=train_gen, steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_gen, validation_steps=STEP_SIZE_VALID, epochs=10)

    # Plot history: MSE -- from https://www.machinecurve.com/index.php/2019/10/08/how-to-visualize-the-training-process-in-keras/#visualizing-the-mse
    plt.plot(history.history['mse'], label='MSE (training data)')
    plt.plot(history.history['val_mse'], label='MSE (validation data)')
    plt.title('MSE for Steering Angle Prediction')
    plt.ylabel('MSE value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper right")
    plt.show()


df = prep.load_dataset_dataframe("matches.csv")
train_gen, valid_gen = prep.get_dataset_generators_from_dataframe(df, "images", "filename", "angle")
fit_model(train_gen, valid_gen)