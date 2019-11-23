import numpy as np
import os
from keras.layers import Input, Conv2D, Conv2DTranspose, Cropping2D, Concatenate, Dropout, Activation, BatchNormalization, ZeroPadding2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.models import Model, load_model
from src import utils
from keras.applications.densenet import DenseNet121

def convblock1(Input_Net, num_filters, block_index, pad='valid', stride=(2, 2)):

	conv = Conv2D(num_filters, kernel_size=7, padding=pad, strides=stride, use_bias=False, name='conv{}/conv'.format(block_index))(Input_Net)
	bn = BatchNormalization(name='conv{}/bn'.format(block_index))(conv)
	relu = Activation('relu', name='conv{}/relu'.format(block_index))(bn)
	return relu

def convblock2(Input_Net, num_filters, block_index, layer_index, pad='same', stride=(1, 1)):

	bn = BatchNormalization(name='conv{}_block{}_{}_bn'.format(block_index, layer_index, 0))(Input_Net)
	relu = Activation('relu', name='conv{}_block{}_{}_relu'.format(block_index, layer_index, 0))(bn)
	conv = Conv2D(num_filters, kernel_size=1, padding=pad, strides=stride, use_bias=False, name='conv{}_block{}_{}_conv'.format(block_index, layer_index, 1))(relu)
	net = BatchNormalization(name='conv{}_block{}_{}_bn'.format(block_index, layer_index, 1))(conv)
	net = Activation('relu', name='conv{}_block{}_{}_relu'.format(block_index, layer_index, 1))(net)
	net = Conv2D(32, kernel_size=3, padding=pad, strides=stride, use_bias=False, name='conv{}_block{}_{}_conv'.format(block_index, layer_index, 2))(net)
	return net

def transitionblock(Input_Net, num_filters, block_index):

	net = BatchNormalization(name='pool{}_bn'.format(block_index))(Input_Net)
	net = Activation('relu', name='pool{}_relu'.format(block_index))(net)
	net = Conv2D(num_filters, kernel_size=1, use_bias=False, name='pool{}_conv'.format(block_index))(net)
	net = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool{}_pool'.format(block_index))(net)
	return net

def denseblock(Input_Net, num_filters, growth_rate, block_index, block_size, pad='same', stride=1):

	concat = Input_Net
	for i in range(block_size):

		net = convblock2(concat, 128, block_index, i)
		concat = Concatenate(axis=3, name='conv{}_block{}_concat'.format(block_index, (i + 1)))([(net), (concat)])
		num_filters = num_filters + growth_rate

	return concat, num_filters

def convtransposeblock(Input_Net, num_filters, block_index, crop, pad='valid'):

	kernel_size = (2**block_index*2, 2**block_index*2)
	strides = (2**block_index, 2**block_index)

	side = Conv2DTranspose(16, kernel_size=kernel_size, strides=strides, padding=pad, kernel_initializer=utils.get_std_init(), trainable=False, name='side{}/ConvTranspose'.format(block_index))(Input_Net)

	side = Cropping2D(cropping=crop, name='side{}/Crop'.format(block_index))(side)
	return side

def densenet121(blocks_sizes=[6, 12, 24, 16], growth_rate=32, trainable=True):

	croplist = [(0, 0), (4, 5), (8, 9), (16, 13), (32, 21)]
	filters = 64
	block_index = 1
	Input_image = Input(shape=(480, 854, 3), name='Input')
	net1 = ZeroPadding2D(padding=(3, 3))(Input_image)
	net1 = convblock1(net1, filters, block_index)

	sides = [convtransposeblock(net1, 16, block_index, croplist[block_index - 1], pad='same')]
	
	net1 = ZeroPadding2D(padding=(2, 2))(net1)
	net1 = MaxPooling2D(pool_size=(3, 3), strides=2, name='pool{}'.format(block_index))(net1)
	block_index += 1

	for i in range(len(blocks_sizes) - 1):

		net1, filters = denseblock(net1, filters, growth_rate, block_index, blocks_sizes[i])
		filters = np.int32(filters/2)
		sides.append(convtransposeblock(net1, 16, block_index, croplist[block_index - 1]))
		net1 = transitionblock(net1, filters, block_index)
		block_index += 1


	net1, filters = denseblock(net1, filters, growth_rate, block_index, blocks_sizes[len(blocks_sizes) -1])
	net1 = BatchNormalization(name='bn')(net1)
	net1 = Activation('relu', name='relu')(net1)
	sides.append(convtransposeblock(net1, 16, block_index, croplist[block_index - 1]))
	# Concatenate outputs
	fusion = Concatenate(axis=3)(sides)

	final_conv = Conv2D(1, (1, 1), strides=(1, 1), padding='valid', activation='sigmoid', kernel_initializer=utils.get_std_init(), use_bias=False)(fusion)

	final_model = Model(Input_image, final_conv)
	return final_model

def initialize_app(net):

	Input_image = Input(shape=(480, 854, 3), name='Input')
	weights = DenseNet121(include_top=False, weights='imagenet', input_tensor=Input_image, input_shape=(480, 854, 3), pooling=None)

	for i in range(len(weights.layers)):
		net.layers[i].set_weights(weights.layers[i].get_weights())
	
	return net

def save_net(net, train_set, path, ckpt):

	file_name = 'DenseVOS_{}_ckpt_{}.h5'.format(train_set, ckpt)
	net.save_weights(filepath=os.path.join(path, file_name))

def save_architecture(net, path):
    json_net = net.to_json()
    with open(file=os.path.join(path, 'DenseVOS_Arch.json'), mode='w') as json_file:
        json_file.write(json_net)