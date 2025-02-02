import numpy
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Conv2D
from keras.layers import Conv1D
from keras.layers.convolutional import MaxPooling2D, MaxPooling1D
from keras.utils import np_utils
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
import keras

import load_data

K.set_image_dim_ordering("tf")
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train) = load_data.data_set

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype("float32")
# X_test = X_test.astype('float32')
X_train = X_train / 255.0
# X_test = X_test / 255.0
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
num_classes = y_train.shape[1]

# Create the model
model = Sequential()
model.add(
    Conv2D(
        32,
        (3,1),
        input_shape=(64, 64),
        padding="same",
        activation="relu",
        kernel_constraint=maxnorm(3),
    )
)
model.add(Dropout(0.2))
model.add(
    Conv2D(32, (3,1), activation="relu", padding="same", kernel_constraint=maxnorm(3))
)
model.add(MaxPooling1D(pool_size=(2)))
model.add(Flatten())
model.add(Dense(512, activation="relu", kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))

# Compile model
epochs = 75
lrate = 0.01
decay = lrate / epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
print(model.summary())
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir="./logs",
        histogram_freq=0,
        batch_size=32,
        write_graph=True,
        write_grads=False,
        write_images=True,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None,
    )
]

# Fit the model
model.fit(
    X_train, y_train, epochs=epochs, batch_size=32, shuffle=True, callbacks=callbacks
)

# Final evaluation of the model
scores = model.evaluate(X_train, y_train, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))

# serialize model to JSONx
model_json = model.to_json()
with open("model/model_face.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model/model_face.h5")
print("Saved model to disk")
