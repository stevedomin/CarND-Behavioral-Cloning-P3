from os.path import join

import cv2
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, BatchNormalization
from keras.layers import Conv2D, Cropping2D
from keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

dataset_root_path = './datasets'
datasets = ['data', 'udacity-data', 'data-recovery', 'data-track2']

def load_dataset(dataset):
	log_file = join(dataset_root_path, dataset, 'driving_log.csv')
	df =  pd.read_csv(log_file, header=None)
	df[7] = dataset
	return df

def load_datasets(datasets):
	dfs = (load_dataset(dataset) for dataset in datasets)
	df = pd.concat(dfs, axis=0, ignore_index=True)
	return df.values

def load_image(dataset, source_path):
	filename = source_path.split('/')[-1]
	current_path = join(dataset_root_path, dataset, 'IMG', filename)
	return cv2.imread(current_path)

def preprocess_sample(sample, images, measurements):
	correction = 0.2
	steering_center = float(sample[3])
	steering_left = steering_center + correction
	steering_right = steering_center - correction

	# The datasets are heavily skewed towards neutral steering angle so we drop
	# some of these images 
	if abs(steering_center) < 0.05 and np.random.random_sample() <= 0.26:
		return

	image_center = load_image(sample[7], sample[0])
	image_center_flipped = np.fliplr(image_center)
	image_left = load_image(sample[7], sample[1])
	image_left_flipped = np.fliplr(image_left)
	image_right = load_image(sample[7], sample[2])
	image_right_flipped = np.fliplr(image_right)

	images.extend([
		image_center, image_center_flipped,
		image_left, image_left_flipped,
		image_right, image_right_flipped
	])
	measurements.extend([
		steering_center, -steering_center,
		steering_left, -steering_left,
		steering_right, -steering_right,
	])

def preprocess_samples(samples, images, measurements):
	for sample in samples:
		preprocess_sample(sample, images, measurements)

images = []
measurements = []

samples = load_datasets(datasets)
preprocess_samples(samples, images, measurements)

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Cropping2D(cropping=((60, 25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Lambda(lambda img: K.tf.image.resize_images(img, [66, 200])))

model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), strides=(1,1), activation='relu'))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(1164, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(50, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(10, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, epochs=3, shuffle=True)

model.save('model.h5')
