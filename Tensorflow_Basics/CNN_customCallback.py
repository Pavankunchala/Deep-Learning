import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten , Dense , Conv2D, MaxPooling2D


class CustomCallbacks(tf.keras.callbacks.Callback):
    
    def on_epoch_end(self, epoch, logs={}):
        
        if(logs.get('accuracy')>0.998):
            
            print("\n 99% acc reached")
            self.model.stop_training = True

#preprocess the images
def preprocess_images(image_set):
    image_set = image_set.reshape(-1, 28, 28, 1)
    image_set = image_set/255.0
    return image_set


#mnist dataset
mnist = tf.keras.datasets.mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images = preprocess_images(training_images)
test_images = preprocess_images(test_images)


model = Sequential()
model.add(Conv2D(64,(3,3),activation='relu',input_shape = (28,28,1)))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu')) 
model.add(Dense(10 ,activation= 'softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    training_images,
    training_labels,
    batch_size=64,
    epochs=20,
    callbacks=[CustomCallbacks()]
)





