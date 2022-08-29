from distutils.command.config import config
import os
import numpy as np
import keras
from keras.models import Model, load_model
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout, concatenate, Input
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from d_utils import timer


CONFIG ={
    'n_levels' : 2,
    'batch_size' : 64,
    'epochs' : 10,
}

class CNN():
    def __init__(self, nlevels=2):
        self.levels = nlevels
        self.config = CONFIG
        
        im_shape1 = (31, 31, 3)
        im_shape2 = (63, 63, 3)

        input_1 = Input(shape=im_shape1, name="level_1")
        M1 = Conv2D(filters = 60, kernel_size = 4, activation='relu')(input_1)
        M1 = MaxPooling2D(pool_size = 2)(M1)

        input_2 = Input(shape=im_shape2, name = "level_2")
        M2 = Conv2D(filters = 60, kernel_size = 6, activation='relu')(input_2)
        M2 = MaxPooling2D(pool_size = 4)(M2)

        M = concatenate(inputs = [M1, M2])
        
        # M = Conv2D(filters = 30, kernel_size = 4, activation='relu')(M)
        # M = MaxPooling2D(pool_size = 2)(M)
        # M = Dropout(0.5)(M)

        M = Flatten()(M)
        M = Dense(2, activation='softmax', name = "classification")(M)

        self.model = Model(inputs = [input_1, input_2], outputs = M)

    def set_config(self, key, value):
        self.config[key] = value

    @timer
    def train(self, train_gen, validation_gen):

        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=["accuracy"]
        )

        history = self.model.fit(train_gen,
            validation_data=validation_gen,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'], 
            verbose=1,
        )
        return history

    def save(self):
        try:
            os.mkdir('model')
        except:
            pass
        self.model.save('model\\model.h5')

    def predict(self, test_gen): 
        predictions = self.model.predict(test_gen, batch_size=self.config['batch_size'], verbose=1)
        return predictions
    
    def display(self, graph=False):
        self.model.summary()
        if graph:
            try:
                plot_model(self.model, to_file="model\\model.png", show_shapes=True)
            except:
                pass 