import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Dropout, MaxPooling2D, Activation
from keras.models import Model
from keras import Sequential
#from metadata import set_data

#NUM_CLASSES defined in metadata
CLASSES = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise'] #, 'disgust']
NUM_CLASSES = len(CLASSES)

input_shape = (48, 48, 1)

#%%
mlp_all = keras.Sequential([
    keras.Input(shape = input_shape), 
    layers.Flatten(),
    layers.Dense(4000, activation = "sigmoid"),
    #layers.Dropout(0.2), 
    layers.Dense(1000, activation = "sigmoid"),
    #layers.Dropout(0.2), 
    layers.Dense(100, activation = "sigmoid"),
    layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)

mlp_all.compile(loss="categorical_crossentropy", 
              optimizer="adam", #keras.optimizers.Adam(learning_rate=0.01), 
              metrics=["accuracy"])


conv_all = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(256, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(256, kernel_size=(3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(256, activation="sigmoid"),
        layers.Dense(7, activation="softmax"),
    ]
)
conv_all.compile(loss="categorical_crossentropy", 
              optimizer="adam", #keras.optimizers.Adam(learning_rate=0.01), 
              metrics=["accuracy"])


#%%
mlp_4 = keras.Sequential([
    keras.Input(shape = input_shape), 
    layers.Flatten(),
    layers.Dense(4000, activation = "sigmoid"),
    #layers.Dropout(0.2), 
    layers.Dense(1000, activation = "sigmoid"),
    #layers.Dropout(0.2), 
    layers.Dense(100, activation = "sigmoid"),
    layers.Dense(6, activation="softmax"),
    ]
)

mlp_4.compile(loss="categorical_crossentropy", 
              optimizer="adam", #keras.optimizers.Adam(learning_rate=0.01), 
              metrics=["accuracy"])


#%%
conv_3 = keras.Sequential(
    [
        keras.Input(shape=(48, 48, 1)),
        layers.Conv2D(18, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(24, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        #layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(24, activation="sigmoid"),
        layers.Dense(6, activation="softmax"),
    ]
)
conv_3.compile(loss="categorical_crossentropy", 
              optimizer="adam", #keras.optimizers.Adam(learning_rate=0.01), 
              metrics=["accuracy"])
#%%
conv_4 = keras.Sequential(
    [
        keras.Input(shape=(48, 48, 1)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="sigmoid"),
        layers.Dense(6, activation="softmax"),
    ]
)
conv_4.compile(loss="categorical_crossentropy", 
              optimizer="adam", #keras.optimizers.Adam(learning_rate=0.01), 
              metrics=["accuracy"])




#%%

conv_4_regularized = keras.Sequential(
    [
        keras.Input(shape=(48, 48, 1)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_regularizer =keras.regularizers.l2( l=0.01)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_regularizer =keras.regularizers.l2( l=0.01)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_regularizer =keras.regularizers.l2( l=0.01)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="sigmoid"),
        layers.Dense(6, activation="softmax"),
    ]
)
conv_4_regularized.compile(loss="categorical_crossentropy", 
              optimizer=keras.optimizers.Adam(learning_rate=0.0003), 
              metrics=["accuracy"])

#%%
from keras.optimizers import Adam,SGD,RMSprop


no_of_classes = 6

model = Sequential()

#1st CNN layer
model.add(Conv2D(64,(3,3),padding = 'same',input_shape = (48,48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

#2nd CNN layer
model.add(Conv2D(128,(5,5),padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout (0.25))

#3rd CNN layer
model.add(Conv2D(512,(3,3),padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout (0.25))



#4th CNN layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

#Fully connected 1st layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))


# Fully connected layer 2nd layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(no_of_classes, activation='softmax'))


opt = Adam(lr = 0.0001)
model.compile(optimizer=opt,loss='categorical_crossentropy', metrics=['accuracy'])



#%%
resnet = keras.applications.ResNet50(include_top = False, input_shape=(48, 48, 3))

flat1 = layers.Flatten()(resnet.layers[-1].output)
class1 = layers.Dense(1024, activation='relu')(flat1)
output = layers.Dense(6, activation='softmax')(class1)

pretrained_resnet = Model(inputs=resnet.inputs, outputs=output)

pretrained_resnet.compile(loss="categorical_crossentropy", 
              optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              metrics=["accuracy"])




#%%

eff=keras.applications.EfficientNetB2(input_shape=(48, 48, 3),include_top=False)

#for layer in eff.layers:
#    layer.trainable=False
    
x=layers.Flatten()(eff.output)

prediction=layers.Dense(6,activation='softmax')(x)

pretrained_eff=Model(inputs=eff.input,outputs=prediction)

pretrained_eff.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])




















