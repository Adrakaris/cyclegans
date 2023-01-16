
from typing import Any
from matplotlib.pyplot import axis
import tensorflow as tf
keras = tf.keras 

from enum import Enum 

from keras.layers.convolutional import Conv2D
from keras.layers import BatchNormalization, LeakyReLU, ReLU, UpSampling2D, Dropout, Concatenate
from tensorflow_addons.layers import InstanceNormalization

class Norm(Enum):
    BN = "bn"
    IN = "in" 
    NONE = "none"


def convLeakyRelu(x, filters:int, kernel:int=4, stride:int=2, norm:Norm=Norm.IN, initialiser:Any="random_normal"):
    """Leaky Relu convolution with normalisation for discriminator"""
    y = Conv2D(
        filters=filters,
        kernel_size=kernel,
        strides=stride,
        padding="same",
        kernel_initializer=initialiser
    )(x)
    if norm == Norm.BN:
        y = BatchNormalization()(y)
    elif norm == Norm.IN:
        y = InstanceNormalization(axis=-1, center=False, scale=False)(y)
    y = LeakyReLU(0.2)(y)
    return y
    

def convBlock(x, filters:int, kernel:int, stride:int=2, norm:Norm=Norm.IN, initialiser:Any="random_normal"):
    y = Conv2D(
        filters=filters,
        kernel_size=kernel,
        strides=stride,
        padding="same",
        kernel_initializer=initialiser
    )(x)
    if norm == Norm.BN:
        y = BatchNormalization()(y)
    elif norm == Norm.IN:
        y = InstanceNormalization(axis=-1, center=False, scale=False)(y)
    y = ReLU()(y)
    return y

def convUpsampleUnet(x, skip, filters:int, kernel:int, antistride=2, norm=Norm.IN, initialiser:Any="random_normal", dropoutRate:int=0):
    y = UpSampling2D(size=antistride)(x)
    y = Conv2D(filters=filters, kernel_size=kernel, strides=1, padding="same", kernel_initializer=initialiser)(y)
    if norm == Norm.IN:
        y = InstanceNormalization(axis=-1, center=False, scale=False)(y)
    elif Norm == Norm.BN:
        y = BatchNormalization()(y)
    y = ReLU()(y)
    if dropoutRate:
        y = Dropout(dropoutRate)(y)
    y = Concatenate()([y, skip])
    return y
