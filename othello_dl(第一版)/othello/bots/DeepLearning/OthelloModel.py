from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Reshape,
    GlobalAveragePooling2D,
    Dense,
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    BatchNormalization,
    Activation,
    add,
    Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import numpy as np, os

class OthelloModel():
    def __init__(self, input_shape=(8,8)):
        self.model_name='model_'+( 'x'.join(list(map(str, input_shape))) )+'.h5'
        self.input_boards = Input(shape=input_shape)

        x_image = Reshape(input_shape+(1,))(self.input_boards)
        resnet_v12 = self.resnet_v1(inputs = x_image, num_res_blocks = 2)

        gap1 = Flatten()(resnet_v12)
        self.pi = Dense(input_shape[0]*input_shape[1], activation='softmax', name='pi')(gap1)

        self.model = Model(inputs=self.input_boards, outputs=[self.pi])
        self.model.compile(loss=['categorical_crossentropy'], optimizer=Adam(0.002))
        print(self.model.summary())
        
    def predict(self, board):
        return self.model.predict(np.array([board]).astype('float32'))[0]
    
    def fit(self, data, batch_size, epochs):
        input_boards, target_policys = list(zip(*data))
        input_boards = np.array(input_boards)
        target_policys = np.array(target_policys)
        self.model.fit(x = input_boards, y = [target_policys], batch_size = batch_size, epochs = epochs)
    
    def set_weights(self, weights):
        self.model.set_weights(weights)
    
    def get_weights(self):
        return self.model.get_weights()
    
    def save_weights(self):
        self.model.save_weights('othello/bots/DeepLearning/models/'+self.model_name)
    
    def load_weights(self):
        self.model.load_weights('othello/bots/DeepLearning/models/'+self.model_name)
    
    def reset(self, confirm=False):
        if not confirm:
            raise Exception('this operate would clear model weight, pass confirm=True if really sure')
        else:
            try:
                os.remove('othello/bots/DeepLearning/models/'+self.model_name)
            except:
                pass
        print('cleared')
        
    
    def resnet_v1(self, inputs, num_res_blocks):
        x = inputs        
        for i in range(2):
            if (i == 0):
                resnet = self.resnet_layer(inputs = x, num_filter = 128, strides=2)    
                resnet = self.resnet_layer(inputs = resnet, num_filter = 128, activation = None)
                resnet = self.resnet_layer(inputs = resnet, num_filter = 128, activation = None)
            else:
                resnet = self.resnet_layer(inputs = x, num_filter = 128)    
                resnet = self.resnet_layer(inputs = resnet, num_filter = 128, activation = None)
                resnet = self.resnet_layer(inputs = resnet, num_filter = 128, activation = None)
            if(i == 0):
                x = self.resnet_layer(inputs = x, num_filter = 128, strides=2)   
            resnet = self.max_pool(inputs = resnet)
            resnet = add([resnet, x])
            resnet = Activation('relu')(resnet)
            x = resnet

        for i in range(2):
            if (i == 0):
                resnet = self.resnet_layer(inputs = x, num_filter = 256, strides=2)    
                resnet = self.resnet_layer(inputs = resnet, num_filter = 256, activation = None)
                resnet = self.resnet_layer(inputs = resnet, num_filter = 256, activation = None)
            else:
                resnet = self.resnet_layer(inputs = x, num_filter = 256)    
                resnet = self.resnet_layer(inputs = resnet, num_filter = 256, activation = None)
                resnet = self.resnet_layer(inputs = resnet, num_filter = 256, activation = None)
            if(i == 0):
                x = self.resnet_layer(inputs = x, num_filter = 256, strides=2) 
            resnet = self.average_pool(inputs = resnet)
            resnet = add([resnet, x])
            resnet = Activation('relu')(resnet)
            x = resnet
        
        return x

    def resnet_layer(self, inputs, num_filter = 16, kernel_size = 3, strides = 1, activation = 'relu', batch_normalization = True, conv_first = True, padding = 'same'):
        
        conv = Conv2D(num_filter, 
                      kernel_size = kernel_size, 
                      strides = strides, 
                      padding = padding,
                      use_bias = False,  
                      kernel_regularizer = l2(1e-4))
        
        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization(axis=3)(x)
            if activation is not None:
                x = Activation(activation)(x)
            
        else:
            if batch_normalization:
                x = BatchNormalization(axis=3)(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x    

    def max_pool(self, inputs, pool_size=(2, 2), strides = 1, padding = 'same'):

        pool = MaxPooling2D(pool_size = pool_size,
                            strides = strides,
                            padding = padding)

        x = pool(inputs)
        return x 

    def average_pool(self, inputs, pool_size=(2, 2), strides = 1, padding = 'same'):

        pool = AveragePooling2D(pool_size = pool_size,
                            strides = strides,
                            padding = padding)

        x = pool(inputs)
        return x