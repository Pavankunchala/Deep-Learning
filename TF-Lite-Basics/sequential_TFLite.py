import tensorflow as tf
import pathlib
from tensorflow.keras.layers import Dense 
from tensorflow.keras.models import Sequential


x =  [-1,0,1 ,2,3,4]

y = [-3,-1,1,3,5,7]

model = Sequential()

model.add(Dense(units  =1, input_shape = [1]))

model.compile(loss='mean_squared_error', optimizer  ='sgd')

model.fit(x,y,epochs =500)

export_dir = 'Sequentail_saved_model/1'

tf.saved_model.save(model , export_dir)

#Lets convert the model 
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)

tfLite_model = converter.convert()

#save the model 
tf_lite_model_file = pathlib.Path('Sequentialmodel.tflite')
tf_lite_model_file.write_bytes(tfLite_model)





