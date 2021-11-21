#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:08:17 2019

@author: hazans1
"""

import numpy as np
import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, BatchNormalization,LeakyReLU, Input
from keras.models import Model
from keras import backend as Ke

experts_inputs = Input(shape=(Noisy.shape[1],))

x = Dense(512, name='hidden1',activation='elu')(experts_inputs)
x = BatchNormalization()(x)

x = Dense(512,name='hidden2',activation='elu')(x)
x = BatchNormalization()(x)

x = Dense(512,name='hidden3',activation='elu')(x)
x = BatchNormalization()(x)

SPP = Dense(257, activation='sigmoid',name='spp')(x)

est_spec=Dense(257, activation='linear',name='est_spec')(x)

model=Model(inputs=experts_inputs,outputs=[SPP,est_spec])

model.compile(loss={'spp': "binary_crossentropy", 'est_spec': 'mean_squared_error'},optimizer="ADAM")
early_stopping = EarlyStopping(monitor='loss', patience=3)
model.fit(Noisy, [Targets_IRM,Targets_clean_log_spec], epochs=50, batch_size=252, callbacks=[early_stopping], verbose=0,validation_split=0.2)


def my_loss(y_pred,y_est):
    
    
    
    
    return bce+mse