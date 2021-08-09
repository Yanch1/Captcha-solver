import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from numpy.lib.financial import rate
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D 
from keras.layers.core import Flatten, Dense
from keras.layers import Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from utils import resize_to_fit
from pathlib import Path
from tqdm import tqdm


LETTER_IMAGES_FOLDER = Path("dataset/abi/")
MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"


# initialize the data and labels
data = []
labels = []

# loop over the input images
i = 0
for image_file in paths.list_images(LETTER_IMAGES_FOLDER):
    if i%1000 == 0:
        print(f"{i} images processed")
    file_name = str(image_file).split(sep='\\')[2]

    # Load the image and convert it to grayscale
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # add dim
    image = np.reshape(image, (image.shape[0], image.shape[1], 1))
    
    #image = np.expand_dims(image, axis=2)

    # Grab the name of the letter
    label = file_name[0]

    # Add the letter image and it's label to our training data
    data.append(image)
    labels.append(label)
    i+=1
print(f'{i}/{i}')
print("Finished preparing images and labels")

# scale the raw pixel intensities to the range [0, 1] (this improves training)
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)


# Split the training data into separate train and test sets
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.1, random_state=42)

# Convert the labels (letters) into one-hot encodings that Keras can work with
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)


# Save the mapping from labels to one-hot encodings.
# We'll need this later when we use the model to decode what it's predictions mean
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)



# Build the neural network
# ----------------------------------------------------------------------------------------
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(80, 215, 1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Dropout(rate=0.3))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(36, activation='softmax'))
 
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

model.summary()

# Start training
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=128, epochs=120, verbose=1,) # callbacks=[model_checkpoint_callback]

model.save(MODEL_FILENAME)
