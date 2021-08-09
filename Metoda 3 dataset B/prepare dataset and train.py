import os
from utils import gen_dataset, load_data, matprint, num_alphabet
from dotenv import dotenv_values

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from keras.models import Sequential, Model

config = dotenv_values('.env')

# Load config 
BATCH_SIZE = int(config['BATCH_SIZE'])
NUM_OF_LETTERS = int(config['NUM_OF_LETTERS'])
EPOCHS = int(config['EPOCHS'])
IMG_ROWS = int(config['IMG_ROWS'])
IMG_COLS = int(config['IMG_COLS'])

# Non-configs
PATH = os.getcwd()
DATA_PATH = os.path.join(PATH, 'data')


# Generate dataset
gen_dataset(DATA_PATH, 7 * 10000 , NUM_OF_LETTERS, IMG_COLS, IMG_ROWS)

# Load dataset
x_train, y_train, x_test, y_test = load_data(DATA_PATH)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# transpose matrix
s_train = []
s_test = []
for i in range(NUM_OF_LETTERS):
    s_train.append(y_train[:, i, :])
    s_test.append(y_test[:, i, :])

#prepare model

input_layer = Input((25, 67, 1))
x = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(input_layer)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(filters=48, kernel_size=(5, 5), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Dropout(0.3)(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)

out = [Dense(num_alphabet, name='digit%d' % i, activation='softmax')(x) for i in range(NUM_OF_LETTERS)]
# out = Dense(num_alphabet*5, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=out)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, s_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_data=(x_test, s_test)
                   )


save_dir = os.path.join(PATH, 'saved_models')
model_name = 'trained_model.h5'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print(f'Saved trained model at {model_path}')