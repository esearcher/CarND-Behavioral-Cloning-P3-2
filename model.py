import tensorflow as tf
import numpy as np
import csv
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

# Loads features and labels from certain dataset
def parse_data(log_file):
    # Reads every line
    lines = []
    with open(log_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    # Read ll images and angles
    images = []
    measurements = []
    correction = 0.2
    for line in lines:
        # Center images and angles
        center = line[0]
        images.append(cv2.imread(center)[...,::-1])
        measurements.append(float(line[3]))
        # Left images with angle correction
        left = line[1]
        images.append(cv2.imread(left)[...,::-1])
        measurements.append(float(line[3]) + correction)
        # Right images with angle correction
        right = line[2]
        images.append(cv2.imread(right)[...,::-1])
        measurements.append(float(line[3]) - correction)
    return np.array(images), np.array(measurements)

# Load different datasets
# Driving forward on first track
X_forw, y_forw = parse_data("../data/forward1/driving_log.csv")
# Driving backwards
X_back, y_back = parse_data("../data/backwards/driving_log.csv")
# Recovery from going on the lane
X_recov, y_recov = parse_data("../data/recoveries/driving_log.csv")

# Concatenate all datasets
X = np.concatenate((X_forw,X_back, X_recov))
y = np.concatenate((y_forw,y_back, y_recov))

# Small test to check integrity of the data
print("Steering angle = {}".format(y[0]))
cv2.imshow("TEST", X[0][...,::-1])
cv2.waitKey(0)

# Model definition
model =  Sequential()
# Image is cropped to avoid the upper half
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
# Image normalization
model.add(Lambda(lambda x: x / 255.0 - 0.5))
# Nvidia model defined
model.add(Conv2D(3, kernel_size=(5, 5),
                 activation='relu',
                 strides=(2,2),
                 padding = 'valid'))
model.add(Conv2D(24, kernel_size=(5, 5),
                 activation='relu',
                 strides=(2,2),
                 padding = 'valid'))
model.add(Conv2D(36, kernel_size=(5, 5),
                 activation='relu',
                 strides=(2,2),
                 padding = 'valid'))
model.add(Conv2D(48, kernel_size=(3, 3),
                 activation='relu',
                 padding = 'valid'))
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 padding = 'valid'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# MSE loss function and adam optimizer
model.compile(loss='mse', optimizer='adam')
model.fit(X,y, validation_split=0.2, shuffle=True, nb_epoch=10)

model.save("model.h5")
