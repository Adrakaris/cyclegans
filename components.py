
from typing import Any
from urllib.request import install_opener
from matplotlib.pyplot import axis
import tensorflow as tf
keras = tf.keras 

from enum import Enum 

from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers import BatchNormalization, LeakyReLU, ReLU, UpSampling2D, Dropout, Concatenate, add, Layer, Lambda
from tensorflow_addons.layers import InstanceNormalization
from tensorlayer.layers import DeformableConv2d
from keras.regularizers import l1
# from deformConvAn import DeformableConvLayer
from deformable_conv.deform_layer import DeformableConv2D as ConvOffset2D

class Norm(Enum):
    BN = "bn"
    IN = "in" 
    NONE = "none"


# l = Lambda()
    

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
    """Conv block (conv2d -> norm -> relu)"""
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
    """Reverse convolution using upsampling2D instead of ConvTranspose
    
    Upsample2D -> Conv2D -> Norm -> (optional) Dropout"""
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

def convTransposeBlock(x, filters:int, kernel:int, antistride=2, initialiser:Any="random_normal"):
    """Upsample using regular transposed convolution
    
    Conv2dtranspose -> instancenorm -> relu"""
    y = Conv2DTranspose(filters=filters, kernel_size=kernel, strides=antistride, padding="same", kernel_initializer=initialiser)(x)
    y = InstanceNormalization(axis=-1, center=False, scale=False)(y)
    y = ReLU()(y)
    return y

def resBlock(x, filters:int, kernel:int, initialiser:Any="random_normal"):
    """y is put through two conv layers, and x is skipped across them and added on
    
    note: original book uses reflectionPadding2D, but I do not
    
    Stride is always 1"""
    y = Conv2D(filters=filters, kernel_size=kernel, strides=1, padding="same", kernel_initializer=initialiser)(x)
    y = InstanceNormalization(axis=-1, center=False, scale=False)(y)
    y = ReLU()(y)
    y = Conv2D(filters=filters, kernel_size=kernel, strides=1, padding="same", kernel_initializer=initialiser)(y)
    y = InstanceNormalization(axis=-1, center=False, scale=False)(y)
    return add([y, x])  # x is skip layer

def deformConvBlock(x, inFilters:int, outFilters:int, kernel:int, stride:int=1, initialiser:Any="random_normal", reg_lambda=10):
    """
    Deformable convolution
    
    Using padding "same"
    DeformConv2D -> Conv2D -> InstNorm -> ReLU
    """
    
    
    off = ConvOffset2D(
        filters=inFilters,
        kernel_size=(kernel, kernel),
        strides=1,
        kernel_initializer=initialiser,
        kernel_regularizer=l1(reg_lambda),
        # padding="same"
    )(x)
    y = Conv2D(filters=outFilters,
               kernel_size=kernel,
               strides=(stride, stride),
               padding="same",
               kernel_initializer=initialiser)(off)
    y = InstanceNormalization(axis=-1, center=False, scale=False)(y)
    y = ReLU()(y)
    
    return y
    
