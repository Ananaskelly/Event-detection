from keras.layers import Conv2D, Dropout, BatchNormalization, MaxPooling2D, Input, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy


class CNNModel:
    def __init__(self, config):
        self.config = config

    def get_model(self):
        inp = Input(self.config.shape)
        x = Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), activation='relu')(inp)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(rate=0.3)(x)

        x = Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(rate=0.3)(x)

        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(rate=0.3)(x)
    
        x = Flatten()(x)
        out = Dense(units=self.config.n_classes, activation='softmax')(x)

        model = Model(inputs=inp, outputs=out)
        opt = Adam(self.config.learning_rate)

        model.compile(optimizer=opt, loss=categorical_crossentropy, metrics=['acc'])
        return model
