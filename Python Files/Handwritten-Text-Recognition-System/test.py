from tensorflow.keras.models import load_model
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

np.set_printoptions(linewidth = 200)

filepath = "C:/Users/Constantin/Documents/Ben's shit/GitHub/Handrwitten-Text-Recognition-System/Models/Model from Trial 5"
model = load_model(filepath, compile = True)

image_path = "C:/Users/Constantin/Documents/Ben's shit/GitHub/Handrwitten-Text-Recognition-System/My Handwriting images/processed images"

image = Image.open(image_path + '/' + str(9) + '.jpg')
image2 = Image.open(image_path + '/' + str(10) + '.jpg')

image = np.array(image)
image2 = np.array(image2)
images = [image,image2]
images = np.array(images)
images = images.reshape(images.shape[0],28,28,1)
images = images.astype('float32')
images = images / 255.0
print(images.shape)
'''image = image.astype('float32')
image2 = np.array(image2)
image2 = image2.astype('float32')



reshaped_image1 = image.reshape((28,28))
reshaped_image2 = image2.reshape((28,28))

images = [reshaped_image1, reshaped_image2]
images = np.array(images)
print(images.shape)'''
predict = model.predict(images)
classes = np.argmax(predict,axis =1)
print(predict)
print(classes)