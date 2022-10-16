from distutils.command.config import config
import os
import numpy as np
import keras
from keras.models import Model, load_model
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout, concatenate, Input
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
from d_utils import timer


CONFIG ={
    'n_levels' : 2,
    'shapes' : [(31, 31, 3), (63, 63, 3)],
    'kernel_sizes' : [2, 4],
    'pool_sizes' : [2, 4],
    'filters' : 512,

    'batch_size' : 64,
    'epochs' : 10,
    
}

class CNN():
    def __init__(self, nlevels=2):
        self.levels = nlevels
        self.config = CONFIG
        self.model = self._build_CNN()

    def _build_CNN(self):
        im_shapes = self.config.get('shapes')

        kernel_sizes = self.config.get('kernel_sizes')

        pool_sizes = self.config.get('pool_sizes')

        n_filters = self.config.get('filters')

        M = []
        inputs = []
        for i in range(self.config.get('n_levels')):
            input = Input(shape=im_shapes[i], name=f"level_{i+1}")
            inputs.append(input)
            L = Conv2D(filters = n_filters, kernel_size = kernel_sizes[i], activation='relu')(input)
            L = MaxPooling2D(pool_size = pool_sizes[i])(L)
            M.append(L)

        M = concatenate(inputs = M)
        M = Conv2D(filters = 512, kernel_size = 4, activation='relu')(M)
        M = MaxPooling2D(pool_size = 2)(M)
        M = Dropout(0.2)(M)
        M = Conv2D(filters = 256, kernel_size = 3, activation='relu')(M)
        M = MaxPooling2D(pool_size = 4)(M)
        M = Dropout(0.2)(M)

        M = Flatten()(M)
        M = Dense(2, activation='softmax', name = "classification")(M)

        return Model(inputs = inputs, outputs = M)

    def set_config(self, key, value):
        self.config[key] = value

    @timer
    def train(self, train_gen, validation_gen):
        callback = EarlyStopping(monitor='loss', patience=3)
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=["accuracy"]
        )

        history = self.model.fit(train_gen,
            validation_data=validation_gen,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'], 
            callbacks=[callback],
            verbose=1,
        )
        return history

    def save(self, name):
        try:
            os.mkdir('model')
        except:
            pass
        self.model.save(f"model\\{name}.h5")

    def predict(self, test_gen): 
        predictions = self.model.predict(test_gen, batch_size=self.config['batch_size'], verbose=1)
        return predictions
    
    def display(self, name="", graph=False):
        self.model.summary()
        if graph:
            try:
                plot_model(self.model, to_file=f"model\\{name}.png", show_shapes=True)
            except:
                pass 