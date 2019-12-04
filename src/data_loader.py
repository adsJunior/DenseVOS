from PIL import Image
import os
import numpy as np
import sys
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
        images_array = []
        labels_array = []
        #unfinished
        # Try doing data augmentation using ImageDataGenerator


    #the initializer of the class
    def __init__(self, train_set, test_set, data_aug=True):
        
        self.data_aug = data_aug

        print('Loading files...')
        # setting train_paths
        self.init_train_paths(train_set)
        self.init_test_paths(test_set)

        if self.data_aug:
            print('Performing data augmentation...')
        
