import numpy as np
import h5py
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input, LSTM, LeakyReLU,Reshape, Conv2DTranspose
import tensorflow as tf
from tensorflow.keras.models import Model
import keras.backend as K

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class AdamLearningRateTracker(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        _lr = tf.cast(optimizer.lr, tf.float32)
        _decay = tf.cast(optimizer.decay, tf.float32)
        _iter = tf.cast(optimizer.iterations, tf.float32)

        lr = K.eval(_lr * (1. / (1. + _decay * _iter)))
        print('LR: {:.6f}'.format(lr))


'''
    Implement the loss function in the main text
'''


def permuatation_jaccard_loss(y_true, y_pred, smooth=0):
    y_true = tf.stack([y_true, y_true, y_true, y_true, y_true, y_true, y_true, y_true], axis=1)
    y_pred = tf.stack([y_pred, tf.image.flip_up_down(y_pred), tf.image.flip_left_right(y_pred),
                       tf.image.rot90(y_pred), tf.image.rot90(tf.image.rot90(y_pred)),
                       tf.image.rot90(tf.image.rot90(tf.image.rot90(y_pred))), tf.image.transpose(y_pred),
                       tf.image.transpose(tf.image.rot90(tf.image.rot90(y_pred)))], axis=1)   # All the symmetry operations of a square
    intersection = K.sum(K.abs(y_true * y_pred), axis=[2, 3, 4])
    loss = 1 - (intersection + smooth) / (
                K.sum(K.square(y_true), [2, 3, 4]) + K.sum(K.square(y_pred), [2, 3, 4]) - intersection + smooth)
    loss = tf.reduce_min(loss, axis=-1)

    return loss


'''
    Define the image predictor model
'''

input = Input(shape=(100, 1,))
lstm = LSTM(128, return_sequences=True)(input)
lstm = LSTM(128, return_sequences=True)(lstm)
lstm = LSTM(128)(lstm)
latent = Dense(10)(lstm)
latent = Dense(50)(latent)
decode = LeakyReLU()(latent)
decode = Dense(512)(decode)
decode = LeakyReLU()(decode)
decode = Dense(1024)(decode)
decode = LeakyReLU()(decode)
decode = Dense(5*5*128)(decode)
decode = LeakyReLU()(decode)
decode = Reshape((5, 5, 128))(decode)
decode = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(decode)
decode = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(decode)
decode = Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same")(decode)
output = Conv2DTranspose(1, 2, activation="sigmoid", strides=1, padding="valid")(decode)
model = Model(input, output)
model.summary()
opt = keras.optimizers.Adam(learning_rate=0.001, decay=1e-5)
model.compile(optimizer=opt, loss=permuatation_jaccard_loss)

'''
    Train the model
'''

for i in range(1, 12):
    print('step: ' + str(i))
    f = h5py.File('2D_Matrix_Eig_5_polygon_Mathematica_no_reflection_filled_' + str(i) + '.h5', 'r')
    Matrix = f['Matrix'][()]
    EigValue = f['EigValue'][()]
    Matrix = np.transpose(Matrix)
    EigValue = np.transpose(EigValue)
    Matrix = np.expand_dims(Matrix, -1)
    EigValue = np.expand_dims(EigValue, -1)
    Diff = 300 * np.concatenate(
        (np.reshape(EigValue[:, 0:1, :], (100000, 1, 1)), EigValue[:, 1:, :] - EigValue[:, 0:-1, :]), axis=1)
    EigValue = Diff
    X_train, X_test, y_train, y_test = train_test_split(EigValue, Matrix, random_state=32, test_size=0.20)

    del Diff
    del Matrix
    del EigValue

    epo = -4 * i + 54
    model.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=2, epochs=epo, callbacks=[AdamLearningRateTracker()])


model.save_weights('ML_LSTM_2D_Matrix_Eig_5_polygon_mathematica_filled_unormalized_no_reflection_spacing_weights_jaccard_full_permutation_leaky_relu_10_latent_kernel_3_3_3.h5')   # Save the weights
