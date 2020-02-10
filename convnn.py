import math
import numpy as np 
import h5py
import matplotlib.pyplot as plt 
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf 
from tensorflow.keras import datasets, layers, models
from tensorflow.python.framework import ops 
from cnn_utils import *
import os

checkpoint_path = 'training_1/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize the data between 0 and 1
train_images, test_images = X_train_orig/255.0, X_test_orig/255.0 
Y_train_orig = Y_train_orig.T 
Y_test_orig = Y_test_orig.T 

plt.figure(figsize=(10,10))
for i in range(25):
	plt.subplot(5,5, i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(train_images[i], cmap=plt.cm.binary)
plt.show()

def createModel():
	model = models.Sequential()
	model.add(layers.Conv2D(64, (3,3), activation='relu', input_shape=(64,64,3)))
	model.add(layers.MaxPooling2D((2,2)))
	model.add(layers.Conv2D(64, (3,3), activation='relu'))
	model.add(layers.MaxPooling2D((2,2)))
	model.add(layers.Conv2D(64, (3,3), activation='relu'))
	model.add(layers.Flatten())
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(6))
	return model

def compileModel(model):
	model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
				  metrics=['accuracy'])
	history = model.fit(train_images, Y_train_orig, epochs = 10, 
						validation_data=(test_images, Y_test_orig),
						callbacks=[cp_callback])
	return history

def plotHistory(history):
	plt.plot(history.history['accuracy'], label='accuracy')
	plt.plot(history.history['val_accuracy'], label='val_accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.ylim([0.5, 1])
	plt.legend(loc='lower right')
	plt.show()
	
def evaluateModel(model):
	test_loss, test_acc = model.evaluate(test_images, Y_test_orig, verbose=2)
	return test_loss, test_acc

def __main__():
	model = createModel()
	history = compileModel(model)
	test_loss, test_acc = evaluateModel(model)
	model.summary()
	plotHistory(history)
	
# Run the main function
__main__() 	