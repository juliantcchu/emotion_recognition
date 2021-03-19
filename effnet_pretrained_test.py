from tensorflow.keras.layers import Lambda,Input,Dense,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import EfficientNetB2

from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob

IMAGE_SIZE=[224,224]
train_path='dataset/images/train'

test_path='dataset/images/test'


train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory(train_path,
                                               target_size=(224,224),batch_size=32,class_mode='categorical')

test_set=test_datagen.flow_from_directory(test_path,
                                               target_size=(224,224),batch_size=32,class_mode='categorical')

#%%
eff=EfficientNetB2(input_shape=IMAGE_SIZE+[3],include_top=False)

for layer in eff.layers:
    layer.trainable=False
  
x=Flatten()(eff.output)

prediction=Dense(7,activation='softmax')(x)

model=Model(inputs=eff.input,outputs=prediction)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


r=model.fit_generator(training_set,validation_data=test_set,
                      epochs=3,
                      steps_per_epoch=len(training_set),
                      validation_steps=len(test_set))

model.save('effnet_pretrained_test')

#%%

eff2=EfficientNetB2(input_shape=IMAGE_SIZE+[3],include_top=False)

  
x=Flatten()(eff2.output)

prediction=Dense(7,activation='softmax')(x)

model_retrained=Model(inputs=eff2.input,outputs=prediction)

model_retrained.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

r2=model_retrained.fit_generator(training_set,validation_data=test_set,
                      epochs=16,
                      steps_per_epoch=len(training_set),
                      validation_steps=len(test_set))

model_retrained.save('effnet_retrained_test')





 