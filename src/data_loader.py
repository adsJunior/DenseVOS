from PIL import Image
import os
import numpy as np
import sys
from keras.applications.densenet import preprocessing
# import keras
# from keras.preprocessing.image import ImageDataGenerator

class Data_Loader:

    # Initialize the train paths of the dataset
    def init_train_paths(self, train_set):
        if isinstance(train_set, list):
            self.train_paths = [tuple(line.split()) for line in train_set]
        elif isinstance(train_set, str):
            with open(train_set, 'r') as file:
                file_lines = file.readlines()
                self.train_paths = [tuple(line.split()) for line in file_lines]
        else:
            self.train_paths = []
    
    # Initialize the test paths of the dataset
    def init_test_paths(self, test_set):
        if isinstance(test_set, list):
            self.test_paths = [line.split()[0] for line in test_set]
        elif isinstance(test_set, str):
            with open(test_set, 'r') as file:
                file_lines = file.readlines()
                self.test_paths = [line.split()[0] for line in file_lines]
        else:
            self.test_paths = []

    def load_train_images(self):
        self.images_train = []
        self.labels_train = []
        print("Loading Davis Dataset...")
        #unfinished
        # Try doing data augmentation using ImageDataGenerator
        for index in enumerate(self.train_paths):
            image = Image.open(self.train_paths[index][0])
            image.load()
            label = Image.open(os.path.join(self.train_paths[index][1]))
            label.load()
            label = label.split()[0]
            if self.data_augmentation:
                if index == 0:
                    print('Performing data augmentation...')
                image_fl = image.transpose(Image.FLIP_LEFT_RIGHT)
                label_fl = label.transpose(Image.FLIP_LEFT_RIGHT)
                image_flt = image_fl.transpose(Image.FLIP_TOP_BOTTOM)
                label_flt = label_fl.transpose(Image.FLIP_TOP_BOTTOM)
                image_flud = image.transpose(Image.FLIP_TOP_BOTTOM)
                label_flud = label.transpose(Image.FLIP_TOP_BOTTOM)
                self.images_train.append(np.array(image_fl, dtype=np.uint8))
                self.labels_train.append(np.array(label_fl, dtype=np.uint8))
                self.images_train.append(np.array(image_flt, dtype=np.uint8))
                self.labels_train.append(np.array(label_flt, dtype=np.uint8))
                self.images_train.append(np.array(image_flud, dtype=np.uint8))
                self.labels_train.append(np.array(label_flud, dtype=np.uint8))
            self.images_train.append(np.array(image, dtype=np.uint8))
            self.labels_train.append(np.array(label, dtype=np.uint8))

    def load_test_images(self):

        self.images_test = []
        for line in self.test_paths:
            image = Image.open(line)
            image.load()
            self.images_test.append(image)

        print('Done dataset initialization')

    #the initializer of the class
    def __init__(self, train_set, test_set, data_augmentation=True):
        
        self.data_augmentation = data_augmentation

        print('Loading files...')
        # setting train_paths
        self.init_train_paths(train_set)
        self.init_test_paths(test_set)
        self.load_train_images()
        self.load_test_images()
        self.train_pointer = 0
        self.test_pointer = 0

    def next_batch(self, step, batch_size):
        '''
        Args:
        step: string to indicate witch step do ou wanna the batch - if it is the train step, send 'train',
        if it is the test step, send 'test'.
        batch_size: number to indicate the size of the batch
        '''
        if (step == 'train'):
            if(self.images_train > self.train_pointer + batch_size):
                #cansei