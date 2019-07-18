import csv
from sklearn.model_selection import train_test_split
import cv2
import random
import numpy as np
import sklearn
from scipy import ndimage
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D
from keras.layers import Cropping2D, Activation, BatchNormalization
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Load csv file with image paths and steering angles into different lists
# for training set and validation set
def load_data(data_path = '/opt/Behavioral-Cloning/Data/', valid_split=0.2):
    lines = []
    with open(data_path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    train_samples, validation_samples = train_test_split(lines,
                                                        test_size=valid_split)

    return train_samples, validation_samples

# Data generator
# Input arguments: list representing info in csv file, path to data
# Returns batches of images and steering angles
def generator(samples, data_path = '/opt/Behavioral-Cloning/Data/',
                batch_size=32):
    num_samples = len(samples)
    while 1:
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            if (offset+batch_size)<num_samples:
                batch_samples = samples[offset:(offset+batch_size)]
            else:
                batch_samples = samples[offset:]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                source_path_center = batch_sample[0]
                filename_center = source_path_center.split('/')[-1]
                center_path = data_path + 'IMG/' + filename_center
                image_center = ndimage.imread(center_path)

                images.append(image_center)
                measurement_center = float(batch_sample[3])
                measurements.append(measurement_center)

                # Augment dataset by flipping images
                images.append(cv2.flip(image_center,1))
                measurements.append(measurement_center*-1.0)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# Define model
def net(loss='mse', optimizer='adam'):
    model = Sequential()

    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

    model.add(Cropping2D(cropping=((70,25),(0,0))))

    model.add(Convolution2D(24,(5,5),strides=(2,2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(36,(5,5),strides=(2,2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(48,(5,5),strides=(2,2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(64,(3,3)))
    model.add(Activation('relu'))

    model.add(Convolution2D(64,(3,3)))
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(Activation('relu'))

    model.add(Dense(50))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('relu'))

    model.add(Dense(1))

    model.compile(loss=loss, optimizer=optimizer)

    return model

# Save a figure with loss over epochs for training and validation sets
def loss_fig(history_object, file_name='model_mse_loss.jpg'):
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig(file_name)

def main():

    # parameters for training
    n_batch = 32
    data_path = '/opt/Behavioral-Cloning/Data/'

    # instantiate training and validation data generators
    train_samples, validation_samples = load_data(data_path=data_path)
    train_generator = generator(train_samples, data_path=data_path,
                                batch_size=n_batch)
    validation_generator = generator(validation_samples, data_path=data_path,
                                    batch_size=n_batch)

    # instatiate model
    model = net()

    # Train model
    checkpoint = ModelCheckpoint("model.h5", save_best_only=True)
    callbacks_list = [checkpoint]
    history_object = model.fit_generator(train_generator,
                        steps_per_epoch=(len(train_samples)//n_batch) + 1,
                        callbacks=callbacks_list,
                        validation_data=validation_generator,
                        validation_steps=(len(validation_samples)//n_batch) + 1,
                        epochs=30)

    # Generate figure of loss over epochs and save
    loss_fig(history_object)

if __name__ == '__main__':
    main()
