from __future__ import division, print_function

import os, json
from glob import glob
import numpy as np
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers.core import Activation, Flatten, Dense, Dropout, Lambda, MaxoutDense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, MaxPooling1D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, Merge, Reshape
from keras.regularizers import l2, activity_l2
from keras import initializations
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image

# In case we are going to use the TensorFlow backend we need to explicitly set the Theano image ordering
from keras import backend as K
K.set_image_dim_ordering('th')



#vgg_mean = np.array([108.64628601, 75.86886597, 54.34005737], dtype=np.float32).reshape((3,1,1))
#vgg_std = np.array([70.53946096, 51.71475228, 43.03428563], dtype=np.float32).reshape((3,1,1))
def vgg_preprocess(x):
    #x = x - vgg_mean
    #x = x / vgg_std
    #return x[:, ::-1] # reverse axis rgb->bgr
    return x

def nothing(x):
    return x

#jeffery mentions that batch_size does not change, hence why reshaping them using constants should work (hope)
def backend_reshape1(x):
    return K.reshape(x, (32, 1028)) #refer to line 296-297 on jeffrey's, batch_size // 2 = 32

def backend_reshape2(x):
    return K.reshape(x, (64, 5)) #refer to line 319-320 on jeffrey's, batch_size = 64

n = 32
momentum = 0.9
leakiness = 0.5
batch_size = 64   #this is a constant for now (???)
class Medium():
    """
        The Medium Imagenet model
    """
    #n = 32
    def __init__(self):
        self.FILE_PATH = 'http://files.fast.ai/models/'
        self.create()
        self.get_classes()


    def get_classes(self):
        """
            Downloads the Imagenet classes index file and loads it to self.classes.
            The file is downloaded only if it not already in the cache.
        """
        fname = 'imagenet_class_index.json'
        fpath = get_file(fname, self.FILE_PATH+fname, cache_subdir='models')
        with open(fpath) as f:
            class_dict = json.load(f)
        self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]
        #self.classes = ["DM0", "DM1", "DM2", "DM3", "DM4"]

    def predict(self, imgs, details=False):
        """
            Predict the labels of a set of images using the VGG16 model.

            Args:
                imgs (ndarray)    : An array of N images (size: N x width x height x channels).
                details : ??
            
            Returns:
                preds (np.array) : Highest confidence value of the predictions for each image.
                idxs (np.ndarray): Class index of the predictions with the max confidence.
                classes (list)   : Class labels of the predictions with the max confidence.
        """
        # predict probability of each class for each image
        all_preds = self.model.predict(imgs)
        # for each image get the index of the class with max probability
        idxs = np.argmax(all_preds, axis=1)
        # get the values of the highest probability for each image
        preds = [all_preds[i, idxs[i]] for i in range(len(idxs))]
        # get the label of the class with the highest probability for each image
        classes = [self.classes[idx] for idx in idxs]
        return np.array(preds), idxs, classes



    def ConvBlock(self):
        #of filters = units on report    kernel_size = filter on report
        model = self.model2
        
        model.add(Convolution2D(n, 7, 7, subsample=(2,2), init='orthogonal', border_mode='same', W_regularizer=l2(0.0002), input_shape=(3,512,512)))
        model.add(LeakyReLU(alpha=leakiness))   # add an advanced activation  https://github.com/fchollet/keras/issues/117
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), input_shape=(32,256,256)))

        model.add(Convolution2D(n, 3, 3, init='orthogonal', border_mode='same', W_regularizer=l2(0.0002), input_shape=(32,127,127)))
        model.add(LeakyReLU(alpha=leakiness))
        model.add(Convolution2D(n, 3, 3, init='orthogonal', border_mode='same', W_regularizer=l2(0.0002), input_shape=(32,127,127)))
        model.add(LeakyReLU(alpha=leakiness))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), input_shape=(32,127,127)))

        model.add(Convolution2D(2 * n, 3, 3, init='orthogonal', border_mode='same', W_regularizer=l2(0.0002), input_shape=(32,63,63)))
        model.add(LeakyReLU(alpha=leakiness))
        model.add(Convolution2D(2 * n, 3, 3, init='orthogonal', border_mode='same', W_regularizer=l2(0.0002), input_shape=(64,63,63)))
        model.add(LeakyReLU(alpha=leakiness))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), input_shape=(64,63,63)))

        model.add(Convolution2D(4 * n, 3, 3, init='orthogonal', border_mode='same', W_regularizer=l2(0.0002), input_shape=(64,31,31)))
        model.add(LeakyReLU(alpha=leakiness))
        model.add(Convolution2D(4 * n, 3, 3, init='orthogonal', border_mode='same', W_regularizer=l2(0.0002), input_shape=(128,31,31)))
        model.add(LeakyReLU(alpha=leakiness))
        model.add(Convolution2D(4 * n, 3, 3, init='orthogonal', border_mode='same', W_regularizer=l2(0.0002), input_shape=(128,31,31)))
        model.add(LeakyReLU(alpha=leakiness))
        model.add(Convolution2D(4 * n, 3, 3, init='orthogonal', border_mode='same', W_regularizer=l2(0.0002), input_shape=(128,31,31)))
        model.add(LeakyReLU(alpha=leakiness))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), input_shape=(128,31,31)))

        model.add(Convolution2D(8 * n, 3, 3, init='orthogonal', border_mode='same', W_regularizer=l2(0.0002), input_shape=(128,15,15)))
        model.add(LeakyReLU(alpha=leakiness))
        model.add(Convolution2D(8 * n, 3, 3, init='orthogonal', border_mode='same', W_regularizer=l2(0.0002), input_shape=(256,15,14)))
        model.add(LeakyReLU(alpha=leakiness))
        model.add(Convolution2D(8 * n, 3, 3, init='orthogonal', border_mode='same', W_regularizer=l2(0.0002), input_shape=(256,15,15)))
        model.add(LeakyReLU(alpha=leakiness))
        model.add(Convolution2D(8 * n, 3, 3, init='orthogonal', border_mode='same', W_regularizer=l2(0.0002), input_shape=(256,15,15)))
        model.add(LeakyReLU(alpha=leakiness))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), input_shape=(256,15,15)))
        model.add(Dropout(0.5))
        
        
        
    def FCBlock(self):
        """
            Adds a fully connected layer of 4096 neurons to the model with a
            Dropout of 0.5

            Args:   None
            Returns:   None
        """
        model = self.model2
        model.add(MaxoutDense(512, nb_feature = 1, init='orthogonal', W_regularizer=l2(0.0002), input_shape=(512,)))
        #model.add(Dense(1024, init='orthogonal', W_regularizer=l2(0.0002)))
        #model.add(Lambda(Maxout))  #possible breaking point



    def create(self):
        """
            Creates the VGG16 network achitecture and loads the pretrained weights.

            Args:   None
            Returns:   None
        """
        #l_in_imgdim = Input(shape=(2))
        model1 = self.model1 = Sequential()
        model2 = self.model2 = Sequential()
        model = self.model = Sequential()

        #change shape
        #model1.add(l_in_imgdim)
        model1.add(Lambda(nothing, input_shape = (2,)))
        model2.add(Lambda(vgg_preprocess, input_shape=(3,512,512)))
        #output_shape=(5,)
        

        model2.add(Convolution2D(n, 7, 7, subsample=(2,2), init='orthogonal', border_mode='same', W_regularizer=l2(0.0002), input_shape=(3,512,512)))
        model2.add(LeakyReLU(alpha=leakiness))   # add an advanced activation  https://github.com/fchollet/keras/issues/117
        model2.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        model2.add(Convolution2D(n, 3, 3, init='orthogonal', border_mode='same', W_regularizer=l2(0.0002)))
        model2.add(LeakyReLU(alpha=leakiness))
        model2.add(Convolution2D(n, 3, 3, init='orthogonal', border_mode='same', W_regularizer=l2(0.0002)))
        model2.add(LeakyReLU(alpha=leakiness))
        model2.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        model2.add(Convolution2D(2 * n, 3, 3, init='orthogonal', border_mode='same', W_regularizer=l2(0.0002)))
        model2.add(LeakyReLU(alpha=leakiness))
        model2.add(Convolution2D(2 * n, 3, 3, init='orthogonal', border_mode='same', W_regularizer=l2(0.0002)))
        model2.add(LeakyReLU(alpha=leakiness))
        model2.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        model2.add(Convolution2D(4 * n, 3, 3, init='orthogonal', border_mode='same', W_regularizer=l2(0.0002)))
        model2.add(LeakyReLU(alpha=leakiness))
        model2.add(Convolution2D(4 * n, 3, 3, init='orthogonal', border_mode='same', W_regularizer=l2(0.0002)))
        model2.add(LeakyReLU(alpha=leakiness))
        model2.add(Convolution2D(4 * n, 3, 3, init='orthogonal', border_mode='same', W_regularizer=l2(0.0002)))
        model2.add(LeakyReLU(alpha=leakiness))
        model2.add(Convolution2D(4 * n, 3, 3, init='orthogonal', border_mode='same', W_regularizer=l2(0.0002)))
        model2.add(LeakyReLU(alpha=leakiness))
        model2.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        model2.add(Convolution2D(8 * n, 3, 3, init='orthogonal', border_mode='same', W_regularizer=l2(0.0002)))
        model2.add(LeakyReLU(alpha=leakiness))
        model2.add(Convolution2D(8 * n, 3, 3, init='orthogonal', border_mode='same', W_regularizer=l2(0.0002)))
        model2.add(LeakyReLU(alpha=leakiness))
        model2.add(Convolution2D(8 * n, 3, 3, init='orthogonal', border_mode='same', W_regularizer=l2(0.0002)))
        model2.add(LeakyReLU(alpha=leakiness))
        model2.add(Convolution2D(8 * n, 3, 3, init='orthogonal', border_mode='same', W_regularizer=l2(0.0002)))
        model2.add(LeakyReLU(alpha=leakiness))
        model2.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model2.add(Dropout(0.5))
        
        model2.add(Flatten())
        model2.add(MaxoutDense(512, nb_feature = 1, init='orthogonal', W_regularizer=l2(0.0002)))
        
        model.add(Merge([model1, model2], mode='concat', concat_axis=1))
        
        #model.add(Reshape((64 // 2, -1)))  #possible breaking point
        model.add(Lambda(backend_reshape1))
        
        model.add(Dropout(0.5))
        model.add(Dense(1024, init='orthogonal', W_regularizer=l2(0.0002)))
        
        model2.add(MaxoutDense(512, nb_feature = 1, init='orthogonal', W_regularizer=l2(0.0002)))
        #model.add(MaxPooling1D(pool_length=2, border_mode='valid'))  #possible breaking point
        
        model.add(Dropout(0.5))
        model.add(Dense(10, init='orthogonal', W_regularizer=l2(0.0002))) #num_units=output_dim * 2 = 10
        #model.add(Reshape((batches, 5)))  #possible breaking point
        model.add(Lambda(backend_reshape2))
        
        model.add(Activation('softmax'))


    def get_batches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=64, class_mode='categorical'):
        """
            Takes the path to a directory, and generates batches of augmented/normalized data. Yields batches indefinitely, in an infinite loop.

            See Keras documentation: https://keras.io/preprocessing/image/
        """
        return gen.flow_from_directory(path, target_size=(512,512),
                class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)


    def ft(self, num):
        """
            Replace the last layer of the model with a Dense (fully connected) layer of num neurons.
            Will also lock the weights of all layers except the new layer so that we only learn
            weights for the last layer in subsequent training.

            Args:
                num (int) : Number of neurons in the Dense layer
            Returns:
                None
        """
        
        model = self.model
        #model.pop()
        #for layer in model.layers: layer.trainable=False
        #model.add(Dense(num, activation='softmax'))
        self.compile()

    def finetune(self, batches):
        """
            Modifies the original VGG16 network architecture and updates self.classes for new training data.
            
            Args:
                batches : A keras.preprocessing.image.ImageDataGenerator object.
                          See definition for get_batches().
        """
        self.ft(batches.nb_class)
        classes = list(iter(batches.class_indices)) # get a list of all the class labels
        
        # batches.class_indices is a dict with the class name as key and an index as value
        # eg. {'cats': 0, 'dogs': 1}

        # sort the class labels by index according to batches.class_indices and update model.classes
        
        for c in batches.class_indices:
            classes[batches.class_indices[c]] = c
        self.classes = classes


    def compile(self, lr=0.003):
        """
            Configures the model for training.
            See Keras documentation: https://keras.io/models/model/
        """
        self.model.compile(optimizer=SGD(lr=lr, momentum=0.9, nesterov=True),
                loss='mean_squared_error', metrics=['accuracy'])


    def fit_data(self, trn, labels,  val, val_labels,  nb_epoch=1, batch_size=64):
        """
            Trains the model for a fixed number of epochs (iterations on a dataset).
            See Keras documentation: https://keras.io/models/model/
        """
        self.model.fit(trn, labels, nb_epoch=nb_epoch,
                validation_data=(val, val_labels), batch_size=batch_size)
        


    def fit(self, batches, val_batches, nb_epoch=1):
        """
            Fits the model on data yielded batch-by-batch by a Python generator.
            See Keras documentation: https://keras.io/models/model/
        """
        self.model.fit_generator(batches, samples_per_epoch=batches.nb_sample, nb_epoch=nb_epoch,
                validation_data=val_batches, nb_val_samples=val_batches.nb_sample, class_weight= class_weight)
        


        
        
    def test(self, path, batch_size=64):
        """
            Predicts the classes using the trained model on data yielded batch-by-batch.

            Args:
                path (string):  Path to the target directory. It should contain one subdirectory 
                                per class.
                batch_size (int): The number of images to be considered in each batch.
            
            Returns:
                test_batches, numpy array(s) of predictions for the test_batches.
    
        """
        test_batches = self.get_batches(path, shuffle=False, batch_size=batch_size, class_mode=None)
        return test_batches, self.model.predict_generator(test_batches, test_batches.nb_sample)