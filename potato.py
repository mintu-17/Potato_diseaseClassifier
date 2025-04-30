!pip install tensorflow


import tensorflow as tf
from tensorflow.keras import models, layers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt


img_size=256
btch_size=32
chanels=3
epoch=20
n_classes = 3

#load a dataset of images stored in a directory structure
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",#folder name where we have leaf img files
    labels='inferred',  # Automatically infer labels from subdirectory names
    label_mode='int',   # Return integer labels
    image_size=(img_size,img_size),
    batch_size=btch_size,
    shuffle=True,
    #seed=123,
    #validation_split=0.2,
    #subset='training'
)

#class_names gives the folders present in the directory
class_names=dataset.class_names
class_names


for img_btch,label_btch in dataset.take(1):  #1 batch 32 imgs
    print(img_btch.shape)  
    #print(img_btch[0].shape) -->for one img
    print(label_btch.numpy())


#3 classes ->early_blight ,late,healty
#we got 3 labels 0,1,2 from the above output


plt.figure(figsize=(20,5))
for img_btch,label_btch in dataset.take(1): 
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(img_btch[i].numpy().astype("uint8"))
        plt.title(class_names[label_btch[i]])
        plt.axis("off")


train_size=0.8
len(dataset)*train_size  #i.e we need 54 samples

train_ds=dataset.take(54)  #first 54
len(train_ds)
val_size=0.1
len(dataset)*val_size  #i.e we need 6 samples
test_ds=dataset.skip(54)
print(len(test_ds))

val_ds=test_ds.take(6)  #first 6 from the remaining  
print(len(val_ds))
# for test remainimg 6 so we use skip starting 6
test_ds=test_ds.skip(6)
print(len(test_ds))


def get_dataset_partitions_df(ds,train_split=0.8,val_split=0.1,test_split=0.1,shuffle=True,shuffle_size=10000):
    ds_size=len(ds)
    if shuffle:
        ds=ds.shuffle(shuffle_size,seed=12)
        
    train_size=int(train_split*ds_size)
    val_size=int(val_split*ds_size)
    
    train_ds=ds.take(train_size)
    val_ds=ds.skip(train_size).take(val_size)
    test_ds=ds.skip(train_size).skip(val_size)
    
    return train_ds,val_ds,test_ds
                          

train_ds,val_ds,test_ds=get_dataset_partitions_df(dataset)

train_ds=train_ds.cache().shuffle(10000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds=val_ds.cache().shuffle(10000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds=test_ds.cache().shuffle(10000).prefetch(buffer_size=tf.data.AUTOTUNE)

resize_and_rescale=tf.keras.Sequential([
    layers.Resizing(256,256),  #resizes if img is not (256,256)
    layers.Rescaling(1.0/255)      #divides or rescales all values by 256
])

data_augmentation=tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2)
])

model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, chanels)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])

model.build(input_shape=(btch_size, img_size, img_size, chanels))

model.summary()


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history=model.fit(train_ds,epochs=epoch,batch_size=btch_size,verbose=1,validation_data=val_ds)

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']


import numpy as np
for img_btch,label_btch in test_ds.take(1): 
    first_img=img_btch[0].numpy().astype('uint8')
    first_label=label_btch[0].numpy()
    
    print("first img to predict")
    plt.imshow(first_img)
    

    
        
    print("actual label:",class_names[first_label])    
    batch_prediction=model.predict(img_btch)
    print("predicted label:",class_names[np.argmax(batch_prediction[0])])


def predict(model,img):
    img_array=tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array=tf.expand_dims(img_array,axis=0)

    predictions=model.predict(img_array)
    predicted_class=class_names[np.argmax(predictions[0])]
    confidence=round(100*(np.max(predictions[0])),2)
    return predicted_class, confidence


for images,labels in test_ds.take(1):
    plt.figure(figsize=(15,15))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        predicted_class,confidence=predict(model,images[i].numpy())
        actual_class=class_names[labels[i]]
        plt.title(f"Actual:{actual_class},\n Predicted:{predicted_class}\n Confidence:{confidence}%.")
        plt.axis("off")          
    #plt.show()
       

import os

# List all files in the directory
files = os.listdir("../models")

# Filter out files that match the pattern (numeric without extension)
model_versions = [int(f.split('.')[0]) for f in files if f.split('.')[0].isdigit()]

# If no numeric versions found, start with 0
if not model_versions:
    model_versions = [0]

# Determine the next model version
model_version = max(model_versions) + 1

# Save the model with a .keras extension
model.save(f"../models/{model_version}.keras")

        


