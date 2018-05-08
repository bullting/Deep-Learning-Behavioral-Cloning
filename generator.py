"""
Created on Sun Aug 13 12:26:13 2017
@author: Kittipong Vadhanakom
"""
import numpy as np
import csv
import cv2
import h5py
import os
import sklearn
import matplotlib as plt

#load data in csv file
log = ['driving_log.csv','driving_log1.csv','driving_log2.csv', 'driving_log_n3.csv']
samples = []
for i in range(len(log)):
    with open(log[i]) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def augmentation(images, angles):
    augmented_images, augmented_angles = [], []
    for image, angle in zip(images, angles):
        augmented_images.append(image)
        augmented_angles.append(angle)
        augmented_images.append(cv2.flip(image,1))
        augmented_angles.append(angle*-1.0)
    return(augmented_images, augmented_angles)

def generator(samples, batch_size=32):
 
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            #for i in range(3):
            for batch_sample in batch_samples:
                name = batch_sample[0].replace(" ", "") 
                image = cv2.imread(name)
                images.append(image)
                angle = float(batch_sample[3])
                angles.append(angle)

                name = batch_sample[1].replace(" ", "") 
                image = cv2.imread(name)
                images.append(image)
                angle = float(batch_sample[3])
                angles.append(angle + 0.2)

                name = batch_sample[2].replace(" ", "")
                image = cv2.imread(name)
                images.append(image)
                angle = float(batch_sample[3])
                angles.append(angle - 0.2)
            
            augmented_images, augmented_angles = augmentation(images, angles)


            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)

validation_generator = generator(validation_samples, batch_size=32)
nbEpoch = 2
ch, row, col = 3, 80, 320

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Cropping2D(cropping=((70,25),(0,0), input_shape=(160,320,3))))
model.add(Lambda(lambda x: x / 127.5 - 1.))

model.add(Convolution2D(24,5,5,subsample=(2,2), activation="relu"))
model.add(Dropout(0.1))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation="relu"))
#model.add(Dropout(0.1))
model.add(Convolution2D(48,3,3, activation="relu"))
model.add(MaxPooling2D())
#model.add(Dropout(0.1))
model.add(Convolution2D(64,3,3,activation="relu"))
#model.add(Dropout(0.1))
model.add(Convolution2D(64,3,3,activation="relu"))

model.add(Flatten())
model.add(Dense(100))
#model.add(Dropout(0.1))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")

history = model.fit_generator(train_generator, samples_per_epoch=(9998),
                              nb_epoch=nbEpoch, validation_data=validation_generator, 
                              nb_val_samples=len(validation_samples))


model.save('model.h5')
exit()




