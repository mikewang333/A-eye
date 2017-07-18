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
from keras.layers.core import Flatten, Dense, Dropout, Lambda, MaxoutDense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2, activity_l2
from keras import initializations
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image

# In case we are going to use the TensorFlow backend we need to explicitly set the Theano image ordering
from keras import backend as K
K.set_image_dim_ordering('th')



vgg_mean = np.array([108.64628601, 75.86886597, 54.34005737], dtype=np.float32).reshape((3,1,1))
vgg_std = np.array([70.53946096, 51.71475228, 43.03428563], dtype=np.float32).reshape((3,1,1))
def vgg_preprocess(x):
    x = x - vgg_mean
    x = x / vgg_std
    #return x[:, ::-1] # reverse axis rgb->bgr
    return x

n = 32
INITIAL_WEIGHTS = [1.36, 14.4, 6.64, 40.2, 49.6]
FINAL_WEIGHTS = [1, 2, 2, 2, 2] 
R = 0.975
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



    def ConvBlock1(self):
        model = self.model
        #of filters = units on report    kernel_size = filter on report
        model.add(Convolution2D(n, 4, 4, subsample=(2,2), border_mode='same', init='orthogonal', W_regularizer=l2(0.0005)))
        model.add(LeakyReLU(alpha=.33))   # add an advanced activation  https://github.com/fchollet/keras/issues/117
        model.add(Convolution2D(n, 4, 4, init='orthogonal', W_regularizer=l2(0.0005)))
        model.add(ZeroPadding2D(padding=(2,2)))
        model.add(LeakyReLU(alpha=.33))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    def ConvBlock2(self):
        model = self.model
        #of filters = units on report    kernel_size = filter on report
        model.add(Convolution2D(2 * n, 4, 4, subsample=(2,2), border_mode='same', init='orthogonal', W_regularizer=l2(0.0005)))
        model.add(LeakyReLU(alpha=.33))   # add an advanced activation  https://github.com/fchollet/keras/issues/117
        model.add(Convolution2D(2 * n, 4, 4, init='orthogonal', W_regularizer=l2(0.0005)))
        model.add(ZeroPadding2D(padding=(2,2)))
        model.add(LeakyReLU(alpha=.33))
        model.add(Convolution2D(2 * n, 4, 4, border_mode='same', init='orthogonal', W_regularizer=l2(0.0005)))
        model.add(LeakyReLU(alpha=.33))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))


    def ConvBlock3(self):
        model = self.model
        #of filters = units on report    kernel_size = filter on report
        model.add(Convolution2D(4 * n, 4, 4, init='orthogonal', W_regularizer=l2(0.0005)))
        model.add(ZeroPadding2D(padding=(2,2)))
        model.add(LeakyReLU(alpha=.33))   # add an advanced activation  https://github.com/fchollet/keras/issues/117
        model.add(Convolution2D(4 * n, 4, 4, border_mode='same', init='orthogonal', W_regularizer=l2(0.0005)))
        model.add(LeakyReLU(alpha=.33))
        model.add(Convolution2D(4 * n, 4, 4, init='orthogonal', W_regularizer=l2(0.0005)))
        model.add(ZeroPadding2D(padding=(2,2)))
        model.add(LeakyReLU(alpha=.33)) 
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    def ConvBlock4(self):
        model = self.model
        #of filters = units on report    kernel_size = filter on report
        model.add(Convolution2D(8 * n, 4, 4, init='orthogonal', W_regularizer=l2(0.0005)))
        model.add(ZeroPadding2D(padding=(2,2)))
        model.add(LeakyReLU(alpha=.33))   # add an advanced activation  https://github.com/fchollet/keras/issues/117
        model.add(Convolution2D(8 * n, 4, 4, border_mode='same', init='orthogonal', W_regularizer=l2(0.0005)))
        model.add(LeakyReLU(alpha=.33))
        model.add(Convolution2D(8 * n, 4, 4, init='orthogonal', W_regularizer=l2(0.0005)))
        model.add(ZeroPadding2D(padding=(2,2)))
        model.add(LeakyReLU(alpha=.33)) 
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    def ConvBlock5(self):
        model = self.model
        #of filters = units on report    kernel_size = filter on report
        model.add(Convolution2D(8 * n, 4, 4, border_mode='same', init='orthogonal', W_regularizer=l2(0.0005)))
        model.add(LeakyReLU(alpha=.33))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        
        
        
    def FCBlock1(self):
        """
            Adds a fully connected layer of 4096 neurons to the model with a
            Dropout of 0.5

            Args:   None
            Returns:   None
        """
        model = self.model
        model.add(Dropout(0.5))
        model.add(Dense(1024, init='orthogonal', W_regularizer=l2(0.0005)))
        model.add(LeakyReLU(alpha=.33))
        model.add(MaxoutDense(512, init='orthogonal', W_regularizer=l2(0.0005)))

 
    def FCBlock2(self):
        """
            Adds a fully connected layer of 4096 neurons to the model with a
            Dropout of 0.5

            Args:   None
            Returns:   None
        """
        model = self.model
        model.add(Dropout(0.5))
        model.add(Dense(1024, init='orthogonal', W_regularizer=l2(0.0005)))
        model.add(LeakyReLU(alpha=.33))
        model.add(MaxoutDense(512, init='orthogonal', W_regularizer=l2(0.0005)))

    def create(self):
        """
            Creates the VGG16 network achitecture and loads the pretrained weights.

            Args:   None
            Returns:   None
        """
        model = self.model = Sequential()

        #change shape
        model.add(Lambda(vgg_preprocess, input_shape=(3,448,448), output_shape=(3,448,448)))


        self.ConvBlock1()
        self.ConvBlock2()
        self.ConvBlock3()
        self.ConvBlock4()
        self.ConvBlock5()

        model.add(Flatten())
        self.FCBlock1()
        self.FCBlock2()
        model.add(Dense(1000, activation='softmax'))


    def get_batches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical'):
        """
            Takes the path to a directory, and generates batches of augmented/normalized data. Yields batches indefinitely, in an infinite loop.

            See Keras documentation: https://keras.io/preprocessing/image/
        """
        return gen.flow_from_directory(path, target_size=(448,448),
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
        model.pop()
        for layer in model.layers: layer.trainable=True
        model.add(Dense(num, activation='softmax'))
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
        
        
    def weight_calculation(self, t, w_0, w_f):
        return ((R ** (t - 1)) * w_0) + ((1 - (R ** (t - 1))) * w_f)


    def fit(self, batches, val_batches, epoch_num, nb_epoch=1):
        """
            Fits the model on data yielded batch-by-batch by a Python generator.
            See Keras documentation: https://keras.io/models/model/
        """
        weights = []
        for i in range(5):
            #print(INITIAL_WEIGHTS[i])
            weights.append(self.weight_calculation(epoch_num, INITIAL_WEIGHTS[i], FINAL_WEIGHTS[i]))
        class_weight = {0: weights[0], 
               1: weights[1],
               2: weights[2],
               3: weights[3],
               4: weights[4]}
        self.model.fit_generator(batches, samples_per_epoch=batches.nb_sample, nb_epoch=nb_epoch,
                validation_data=val_batches, nb_val_samples=val_batches.nb_sample, class_weight= class_weight)
        


        
        
    def test(self, path, batch_size=8):
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