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
import os, pathlib

data_dir = "new_test_images"
IMG_HEIGHT = 64
IMG_WIDTH = 64
AUTOTUNE = tf.data.experimental.AUTOTUNE
CLASS_NAMES = ['one', 'two', 'three', 'four', 'five']
list_ds = tf.data.Dataset.list_files(data_dir+'/*/*')
for f in list_ds.take(5):
	print(f.numpy())

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

def get_label(file_path):
	# convert the path to a list of path components
	parts = tf.strings.split(file_path, os.path.sep)
	# The second to last is the class-directory
	return parts[-2] == CLASS_NAMES

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
	img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
	img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
	return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
	label = get_label(file_path)
    # load the raw data from the file as a string
	img = tf.io.read_file(file_path)
	img = decode_img(img)
	return img, label

def testModel(model):
	test_images = np.array()
	labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
	for image, label in labeled_ds.take(5):
		print("Image shape: ", image.numpy().shape)
		print("Label: ", label.numpy())
		print(image.shape)
		test_images.append(image)
	print(test_images.shape)
	predictions = model.predict(test_images)
	print(predictions)


def __main__():
	model = createModel()
	testModel(model)
	history = compileModel(model)
	test_loss, test_acc = evaluateModel(model)
	model.summary()
	plotHistory(history)



	


# Run the main function
__main__() 	