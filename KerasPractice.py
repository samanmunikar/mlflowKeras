
# coding: utf-8

# In[2]:

import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import math
import h5py


# In[3]:

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def load_dataset():
    train_dataset = h5py.File('datasets/train_happy.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # labels
    
    test_dataset = h5py.File('datasets/test_happy.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) #features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # labels
    
    classes = np.array(test_dataset["list_classes"][:]) # list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1,train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1,test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# In[4]:

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

#Normalize image vectors
X_train = X_train_orig/255
X_test = X_test_orig/255

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print("number of training examples: ", X_train.shape[0])
print("number of test examples: ", X_test.shape[0])
print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("Y_train shape: ", Y_train.shape)
print("Y_test shape: ", Y_test.shape)


# In[5]:

def HappyModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset
        (height, width, channels) as a tuple.  
        Note that this does not include the 'batch' as a dimension.
        If you have a batch like 'X_train', 
        then you can provide the input_shape using
        X_train.shape[1:]
    

    Returns:
    model -- a Model() instance in Keras
    """
    
    X_input = Input(input_shape)
    
    # Zero-Padding: pads the border of X with zeros
    X = ZeroPadding2D((3,3)) (X_input)
    
    # Conv -> BN -> RELU Block applied to X
    X = Conv2D(32, (7,7), strides=(1,1), name="conv0")(X)
    X = BatchNormalization(axis=3, name="bn0")(X)
    X = Activation("relu")(X)
    
    # MAXPOOL
    X = MaxPooling2D((2,2), name="max_pool0")(X)
    
    # Conv -> BN -> RELU Block applied to X
    X = Conv2D(32, (3,3), strides=(1,1), name="conv1")(X)
    X = BatchNormalization(axis=3, name="bn1")(X)
    X = Activation("relu")(X)
    
    # MAXPOOL
    X = MaxPooling2D((2,2), name="max_pool1")(X)
    
    # FLATTEN X(means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation="sigmoid", name="fc")(X)
    
    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs = X, name="HappyModel")
    
    return model


# In[6]:

# Create a model
happyModel = HappyModel(X_train.shape[1:])


# In[7]:

# compile the model
happyModel.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# In[11]:

# Train the model
happyModel.fit(X_train, Y_train, epochs=40, batch_size=16)


# In[12]:

# Evaluate Performance
preds = happyModel.evaluate(X_test, Y_test)

print("Loss = ", preds[0])
print("Test Accuracy = ", preds[1])


# In[13]:

happyModel.summary()






