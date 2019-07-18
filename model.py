import csv
from sklearn.model_selection import train_test_split
import cv2
import random
import numpy as np
import sklearn
from scipy import ndimage

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

if __name__ == '__main__':
    main()
