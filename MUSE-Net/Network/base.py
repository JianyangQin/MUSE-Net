import numpy as np
import tensorflow as tf
from keras.layers import Input,Activation,Dropout,BatchNormalization,AveragePooling2D,GlobalAveragePooling2D, GlobalMaxPooling2D, Multiply, Dense, Permute
from keras.layers import Lambda,Reshape,Concatenate,Add
from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.layers import Layer
import keras.backend as K


# Relu-BN-Conv2D 3x3
def conv_unit0(Fin, Fout, drop, H, W):
    unit_input = Input(shape=(Fin, H, W))

    unit_conv = Activation('relu')(unit_input)
    unit_conv = BatchNormalization()(unit_conv)
    unit_conv = Dropout(drop)(unit_conv)
    unit_output = Conv2D(filters=Fout, kernel_size=(3, 3), padding="same")(unit_conv)
    unit_model = Model(inputs=unit_input, outputs=unit_output)
    print('kernel=(3,3)')
    return unit_model


# Relu-BN-Conv2D 1x1
def conv_unit1(Fin, Fout, drop, H, W):
    unit_input = Input(shape=(Fin, H, W))

    unit_conv = Activation('relu')(unit_input)
    unit_conv = BatchNormalization()(unit_conv)
    unit_conv = Dropout(drop)(unit_conv)
    unit_output = Conv2D(filters=Fout, kernel_size=(1, 1), padding="same")(unit_conv)
    unit_model = Model(inputs=unit_input, outputs=unit_output)
    print('kernel=(1,1)')
    return unit_model


# new resdual block
def Res_plus(F, Fplus, rate, drop, H, W):
    cl_input = Input(shape=(F, H, W))

    cl_conv1A = conv_unit0(F, F - Fplus, drop, H, W)(cl_input)

    if rate == 1:
        cl_conv1B = cl_input
    if rate != 1:
        cl_conv1B = AveragePooling2D(pool_size=(rate, rate), strides=(rate, rate), padding="valid")(cl_input)

    cl_conv1B = Activation('relu')(cl_conv1B)
    cl_conv1B = BatchNormalization()(cl_conv1B)

    plus_conv = Conv2D(filters=Fplus * H * W, kernel_size=(int(np.floor(H / rate)), int(np.floor(W / rate))),
                       padding="valid")

    cl_conv1B = plus_conv(cl_conv1B)

    cl_conv1B = Reshape((Fplus, H, W))(cl_conv1B)

    cl_conv1 = Concatenate(axis=1)([cl_conv1A, cl_conv1B])

    cl_conv2 = conv_unit0(F, F, drop, H, W)(cl_conv1)

    cl_out = Add()([cl_input, cl_conv2])

    cl_model = Model(inputs=cl_input, outputs=cl_out)

    return cl_model