import numpy as np
import pandas as pd

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, MaxPooling2D, Conv2D, Input


#%% Load and reshape data
def load_data(isfloat=False, isNorm=False):
    train = pd.read_csv('fashion-mnist_train.csv')
    test = pd.read_csv('fashion-mnist_test.csv')
    
    train_lbl = train['label'].values
    test_lbl = test['label'].values
    train_lbl_dummy = pd.get_dummies(train_lbl).values
    test_lbl_dummy = pd.get_dummies(test_lbl).values
    train_flat = train.loc[:, train.columns!='label'].values
    test_flat = test.loc[:, test.columns!='label'].values
    train_img = train_flat.reshape((60000, 28, 28))
    test_img = test_flat.reshape((10000, 28, 28))
    
    if isfloat == True:
        train_lbl = train_lbl.astype('float32')
        test_lbl = test_lbl.astype('float32')
        train_lbl_dummy = train_lbl_dummy.astype('float32')
        test_lbl_dummy = test_lbl_dummy.astype('float32')
        train_flat = train_flat.astype('float32')
        test_flat = test_flat.astype('float32')
        train_img = train_img.astype('float32')
        test_img = test_img.astype('float32')
    
    if isNorm == True:
        train_flat = train_flat/255
        test_flat /= 255
        train_img /= 255
        test_img /= 255

    return {'train_lbl': train_lbl, 'test_lbl': test_lbl,
            'train_lbl_dummy':train_lbl_dummy, 'test_lbl_dummy':test_lbl_dummy,
            'train_flat':train_flat, 'test_flat':test_flat,
            'train_img':train_img, 'test_img':test_img}


#%% Dense autoencoder
def dense_autoencoder(data):
    train_flat = data['train_flat']
    test_flat = data['test_flat']
    
    act1 = 'relu'
    act2 = 'sigmoid'    
    epoch = 100
    
#    a = Input(shape=(train_flat.shape[1],))
#    b = Dense(32, activation=act1)(a)
#    c = Dense(train_flat.shape[1], activation=act2)(b)
#    model = Model(inputs=a, outputs=c)
    
    input_shape = (train_flat.shape[1],)
    model = Sequential()
    model.add(Dense(256, activation=act1,input_shape=input_shape))
    model.add(Dense(64, activation=act1,input_shape=input_shape))
    model.add(Dense(32, activation=act1,input_shape=input_shape))
    model.add(Dense(64, activation=act1,input_shape=input_shape))
    model.add(Dense(256, activation=act1,input_shape=input_shape))
    model.add(Dense(train_flat.shape[1], activation=act2))

    model.compile(optimizer='adadelta',
                  loss='binary_crossentropy',
                  metrics=['mae'])
    
    model.fit(x=train_flat,
              y=train_flat,
              epochs=epoch,
              callbacks=None,
              validation_data=(test_flat, test_flat),
              batch_size=256)
    
    model.summary()
    return model


#%% Dense autoencoder
#def cnn_autoencoder(data):
    
    
    
    


























    