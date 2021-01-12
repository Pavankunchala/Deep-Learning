import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten , Dense


class CustomCallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch,logs ={}):
        
        if(logs.get('accuracy')> 0.99):
            
            print("\n 99% acc reached")
            self.model.stop_training = True

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

model = Sequential()

model.add(Flatten(input_shape = (28,28)))
model.add(Dense(units = 256 , activation = 'relu'))
model.add(Dense(units =128, activation  = 'relu'))
model.add(Dense(units =10 ,activation = 'softmax'))

model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(
    x_train,
    y_train,
    epochs=10,
    callbacks=[CustomCallbacks()]
)


