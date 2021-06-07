
#import relavent modules
 
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from extra_keras_datasets import emnist
from tensorflow.keras.models import save_model
import numpy as np
np.set_printoptions(linewidth = 200)


number_of_classes = 27

#load emnist data set
#Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373
(x_train, y_train),(x_test, y_test) = emnist.load_data(type='letters')



#reshape the data to be compatable with the first convolutional layer
x_train = x_train.reshape(x_train.shape[0], 28, 28,1)
x_test = x_test.reshape(x_test.shape[0], 28,28,1)


# Parse numbers as floats
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#normalizing the data
x_train = x_train/255.0
x_test = x_test/255.0


#convert y vectors into categorical targets
y_train = tf.keras.utils.to_categorical(y_train, number_of_classes)
y_test = tf.keras.utils.to_categorical(y_test, number_of_classes)



#plots metrics vs epochs
def plotting_function(epochs, history, metrics):
    plt.figure()
    plt.xlabel("Epoch")
    for j in metrics:
        x = history[j]
        plt.plot(epochs[1:], x[1:], label = j) #plots each curve
    plt.legend()
    plt.show()


#creates and compiles the neural network
'''hyperparameters to change here: # of filters, pooling size, filtersize, dropout rate, and the number or neurons
        in the dense layers'''


def create_RNNModel(learning_rate):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation= 'relu', input_shape=(28,28,1)))

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(64,kernel_size=(3,3), activation='relu' ))

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(256, activation= 'relu'))

    model.add(tf.keras.layers.Dense(number_of_classes, activation= 'softmax')) #change this depending on which data set I am using

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) ,metrics=['accuracy']) #had to change from sparce categorical crossentropy to categorical crossentropy because of an error

    return model

def train_RNN(model, train_features, train_label, no_epochs,validation_split1, batch_size = None,):

    history = model.fit(x = train_features,y = train_label, batch_size = batch_size,
                        epochs = no_epochs, verbose=1, validation_split = validation_split1, shuffle = True)



    #tracks the progression of training
    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist



#hyperparameters
learning_rate = .003
epochs = 15
batch_size = 500
validation_split = 0.2


#initialize the model
my_RNN = create_RNNModel(learning_rate)

#train the model
epochs, hist = train_RNN(my_RNN,x_train, y_train, epochs, validation_split,batch_size)

print(hist)
metrics = ['accuracy', 'val_accuracy']
plotting_function(epochs, hist, metrics)

print("evaluate against test set")
my_RNN.evaluate(x=x_test, y = y_test, batch_size= batch_size)

filepath = "C:/Users/Constantin/Documents/Ben's shit/GitHub/Handrwitten-Text-Recognition-System/Models/Model from Trial 4 with letters dataset"
save_model(my_RNN,filepath)