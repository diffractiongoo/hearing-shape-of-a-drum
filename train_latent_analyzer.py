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
    Load the weights for the trained image predictor model
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

model.load_weights('ML_LSTM_2D_Matrix_Eig_5_polygon_mathematica_filled_unormalized_no_reflection_spacing_weights_jaccard_full_permutation_leaky_relu_10_latent_kernel_3_3_3.h5')


'''
    Define the latent analyzer model
'''

sub_model = Model(inputs=model.inputs, outputs=model.layers[4].output)   # Select the first 4 layers from the original model
dense = Dense(50)(sub_model.output)   # Control the number of hidden layers by commenting out lines
dense = LeakyReLU()(dense)
# dense = Dense(50)(dense)
# dense = LeakyReLU()(dense)
# dense = Dense(50)(dense)
# dense = LeakyReLU()(dense)
out = Dense(10)(dense)

model_2 = Model(inputs=sub_model.inputs, outputs=out)
for layer in model_2.layers[:5]:   # The first 4 layers are not trainable
    layer.trainable = False

opt = keras.optimizers.Adam(learning_rate=0.001, decay=1e-5)
model_2.compile(optimizer=opt, loss='mse')

'''
    Train the latent analyzer model
'''

for i in range(1, 12):
    print('step: ' + str(i))

    # Load the training data #
    f = h5py.File('2D_Matrix_Eig_5_polygon_Mathematica_no_reflection_filled_' + str(i) + '.h5', 'r')
    Vertices = f['Vertices'][()]
    EigValue = f['EigValue'][()]
    Angles = f['Angles'][()]
    Vertices = np.transpose(Vertices)
    EigValue = np.transpose(EigValue)
    Angles = np.transpose(Angles)

    # compute the Weyl parameters #
    EigValue = np.expand_dims(EigValue, -1)
    Diff = 300 * np.concatenate(
        (np.reshape(EigValue[:, 0:1, :], (100000, 1, 1)), EigValue[:, 1:, :] - EigValue[:, 0:-1, :]), axis=1)
    EigValue = Diff
    x = Vertices[:, :5]
    y = Vertices[:, 5:]
    area = np.absolute(1 / 2 * (np.sum(x * np.roll(y, -1, axis=1), axis=1) - np.sum(
        np.roll(x, -1, axis=1) * y, axis=1))) / (4*np.pi)
    area = np.expand_dims(area, axis=-1)
    distance = np.sqrt((x - np.roll(x, -1, axis=1)) ** 2 + (y - np.roll(y, -1, axis=1)) ** 2)
    perimeter = np.expand_dims(np.sum(distance, axis=1) / (4*np.pi), axis=-1)
    angle = np.expand_dims(np.sum(np.pi / Angles - Angles / np.pi, axis=1) / 24, axis=-1)
    ground = np.concatenate((area, perimeter, angle), axis=1)
    # ground = Vertices / 2
    # ground = np.concatenate((distance / 4, Angles / (2*np.pi)), axis=1)

    X_train, X_test, y_train, y_test = train_test_split(EigValue, ground, random_state=32, test_size=0.20)
    model_2.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=2, epochs=10, callbacks=[AdamLearningRateTracker()])

model_2.save_weights('ML_LSTM_2D_Matrix_Eig_5_polygon_mathematica_filled_unormalized_fully_reflected_spacing_weights_jaccard_full_permutation_leaky_relu_10_latent_kernel_one_hidden_50_weyl_network.h5')   # Save the weights

'''
    Test the latent analyzer model
'''

# Load the testing data #
f = h5py.File('2D_Matrix_Eig_5_polygon_Mathematica_no_reflection_filled_12.h5', 'r')
Vertices = f['Vertices'][()]
EigValue = f['EigValue'][()]
Angles = f['Angles'][()]
Vertices = np.transpose(Vertices)
EigValue = np.transpose(EigValue)
Angles = np.transpose(Angles)

# compute the Weyl parameters #
EigValue = np.expand_dims(EigValue, -1)
Diff = 300 * np.concatenate(
    (np.reshape(EigValue[:, 0:1, :], (100000, 1, 1)), EigValue[:, 1:, :] - EigValue[:, 0:-1, :]), axis=1)
EigValue = Diff
x = Vertices[:, :5]
y = Vertices[:, 5:]
area = np.absolute(1 / 2 * (np.sum(x * np.roll(y, -1, axis=1), axis=1) - np.sum(
    np.roll(x, -1, axis=1) * y, axis=1))) / (4*np.pi)
area = np.expand_dims(area, axis=-1)
distance = np.sqrt((x - np.roll(x, -1, axis=1)) ** 2 + (y - np.roll(y, -1, axis=1)) ** 2)
perimeter = np.expand_dims(np.sum(distance, axis=1) / (4*np.pi), axis=-1)
angle = np.expand_dims(np.sum(np.pi / Angles - Angles / np.pi, axis=1) / 24, axis=-1)
ground = np.concatenate((area, perimeter, angle), axis=1)
# ground = Vertices / 2
# ground = np.concatenate((distance / 4, Angles / (2 * np.pi)), axis=1)

prediction = model_2.predict(EigValue)   # Make predictions

error = np.absolute(prediction - ground)   # Compute the absolute error


hf = h5py.File('ML_LSTM_2D_Matrix_Eig_5_polygon_mathematica_filled_unormalized_no_reflection_spacing_weights_jaccard_full_permutation_leaky_relu_10_latent_kernel_3_3_3_weyl_prediction_error.h5', 'a')   # Save the data
hf.create_dataset('10-50-3', data=error)
hf.close()
