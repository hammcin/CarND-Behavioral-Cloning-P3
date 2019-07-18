import csv
from sklearn.model_selection import train_test_split

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

def main():

    # parameters for training
    data_path = '/opt/Behavioral-Cloning/Data/'

    # instantiate training and validation data generators
    train_samples, validation_samples = load_data(data_path=data_path)

if __name__ == '__main__':
    main()
