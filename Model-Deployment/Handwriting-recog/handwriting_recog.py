import numpy as np
from numpy.core.defchararray import mod
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten , Conv2D , MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization

#create a load image file function to explore the datasets

def loadImageFile(fileimage):
    f = open(fileimage, "rb")
    f.read(16)
    pixels = 28*28
    images_arr = []
    
    while True:
        try:
            img = []
            for j in range(pixels):
                pix = ord(f.read(1))
                img.append(pix/255)
                
            images_arr.append(img)
            
        except:
            
            break
    f.close()
    
    image_sets = np.array(images_arr)
    return image_sets

#create a load lable file function

def loadLabelFile(filelable):
    f= open(filelable, 'rb')
    f.read(8)
    
    labels_arr = []
    
    while True:
        row = [0 for x in range(10)]
        try:
            label = ord(f.read(1))
            row[label] = 1
            labels_arr.append(row)
            
        except:
            break
        
    f.close()
    label_sets = np.array(labels_arr)
    return label_sets

train_images = loadImageFile("train-images-idx3-ubyte")
train_labels = loadLabelFile("train-labels-idx1-ubyte")
test_images = loadImageFile("t10k-images-idx3-ubyte")
test_labels = loadLabelFile("t10k-labels-idx1-ubyte")
    
    
x_train = train_images.reshape(train_images.shape[0], 28, 28, 1)
x_test = test_images.reshape(test_images.shape[0], 28, 28, 1)

y_train = train_labels
y_test = test_labels

#lets create the model
model = Sequential()


model.add(Conv2D(32, (3, 3),input_shape = (28,28,1),activation='relu'))
model.add(BatchNormalization(axis = -1))

model.add(Conv2D(32, (3,3),activation='relu'))
model.add(MaxPooling2D(pool_size =(2,2)))
model.add(BatchNormalization(axis = -1))

model.add(Conv2D(64, (3,3),activation='relu'))
model.add(BatchNormalization(axis = -1))

model.add(Conv2D(64, (3,3),activation='relu'))
model.add(MaxPooling2D(pool_size =(2,2)))

model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(512 ,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(10,activation= 'softmax'))

#compiling the model
model.compile(loss = 'categorical_crossentropy',optimizer = 'Adam',metrics = ['accuracy'] )

#fit the model
model.fit(x_train, y_train,
             batch_size=100,
             epochs=10,
             verbose=2,
             validation_split=0.2)

score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#save the model to json format(we are saving the structure)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

#let's save the weights
model.save_weights('weights.h5')




