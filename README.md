# Transfer-Learning-for-binary-classification
## Aim
To Implement Transfer Learning for Horses_vs_humans dataset classification using InceptionV3 architecture.
## Problem Statement

The objective of this project is to build a binary classification model that can distinguish between images of horses and humans using transfer learning. By leveraging a pre-trained InceptionV3 model (trained on ImageNet) as the base, we aim to fine-tune it for our specific task while keeping the pre-trained layers frozen. The task involves:

1)Loading the Horse or Human dataset.<br>
2)Implementing data augmentation to prevent overfitting.<br>
3)Adding custom layers to the pre-trained model to suit the binary classification problem.<br>
4)Using callbacks to stop training once the model achieves an accuracy greater than 97%.<br>
5)Visualizing the performance of the model using training and validation accuracy and loss curves.<br>


![image](https://github.com/user-attachments/assets/c9c5831b-927b-44dd-a168-eac55d4c8582)


## DESIGN STEPS
### STEP 1:
</br>
Load the Pre-Trained Model: Import the InceptionV3 model without the top layers, load its pre-trained weights, freeze all layers to make them non-trainable, and get the output from the 'mixed7' layer.

### STEP 2:
</br>
Build the Custom Model: Add custom layers on top of the pre-trained model, including a Flatten layer, Dense layers, a Dropout layer, and an output layer with a sigmoid activation for binary classification.

### STEP 3:
<br>
Prepare the Datasets: Extract the 'Horse or Human' dataset and validation dataset, then create ImageDataGenerators for data augmentation (for training) and normalization (for validation).

### STEP 4:
<br>
Compile the Model: Compile the model using RMSprop optimizer, binary cross-entropy loss, and accuracy as a metric. Then, set up a custom callback to stop training once 97% accuracy is achieved.

### STEP 5:
<br>
Train and Visualize: Train the model on the training dataset, validate it on the validation dataset, and plot the training and validation accuracy/loss over the epochs with your name and register number in the title.

<br/>

## PROGRAM

#### Import all the necessary Libraries and Packages
```python

import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from os import getcwd
```

#### Import the inception model
```
path_inception = '/content/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'


from tensorflow.keras.applications.inception_v3 import InceptionV3

#### Create an instance of the inception model from the local pre-trained weights

local_weights_file = path_inception

pre_trained_model = InceptionV3(include_top = False,
                                input_shape = (150, 150, 3),
                                weights = None)

pre_trained_model.load_weights(local_weights_file)
```

#### Make all the layers in the pre-trained model non-trainable

```
for layer in pre_trained_model.layers:
  layer.trainable = False
```

#### Print the model summary

```
pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output.shape)
last_output = last_layer.output

```
#### Callback Function

```
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs['accuracy']>0.97:
            self.model.stop_training = True
            print("\nReached 97.0% accuracy so cancelling training!")

```

#### Model Architecture

```
from tensorflow.keras.optimizers import RMSprop


x = tf.keras.layers.Flatten()(last_output)

x = tf.keras.layers.Dense(1024, activation='relu')(x)

x = tf.keras.layers.Dropout(0.2)(x)

x = tf.keras.layers.Dense(1, activation='sigmoid')(x)


model = Model(pre_trained_model.input, outputs = x)

```
#### Compile the Model

```
model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00001),
        loss='binary_crossentropy', # use a loss for binary classification
        metrics=['accuracy']
    )

```
#### Summary of the Model

```
print('Name: Vijay Shankar M      Register Number: 212222040178')
model.summary()
```

#### Get Datasets

```
# Get the Horse or Human dataset
path_horse_or_human = '/content/horse-or-human.zip'
# Get the Horse or Human Validation dataset
path_validation_horse_or_human = '/content/validation-horse-or-human.zip'
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import zipfile

local_zip = path_horse_or_human
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/training')
zip_ref.close()

local_zip = path_validation_horse_or_human
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/validation')
zip_ref.close()

```

#### Training and Validation Files
```
train_dir = '/tmp/training'
validation_dir = '/tmp/validation'

train_horses_dir = os.path.join(train_dir, 'horses')
train_humans_dir = os.path.join(train_dir, 'humans')
validation_horses_dir = os.path.join(validation_dir, 'horses')
validation_humans_dir = os.path.join(validation_dir, 'humans')

train_horses_fnames = os.listdir(train_horses_dir)
train_humans_fnames = os.listdir(train_humans_dir)
validation_horses_fnames = os.listdir(validation_horses_dir)
validation_humans_fnames = os.listdir(validation_humans_dir)

print(len(train_horses_fnames))
print(len(train_humans_fnames))
print(len(validation_horses_fnames))
print(len(validation_humans_fnames))

```

#### Data Augementation & ImageDataGenerator

```
train_datagen = ImageDataGenerator(rescale = 1/255,
                                  height_shift_range = 0.2,
                                  width_shift_range = 0.2,
                                  horizontal_flip = True,
                                  vertical_flip = True,
                                  rotation_range = 0.4,
                                  shear_range = 0.1,
                                  zoom_range = 0.3,
                                  fill_mode = 'nearest'
                                  )

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale = 1/255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                   target_size = (150, 150),
                                                   batch_size = 20,
                                                   class_mode = 'binary',
                                                   shuffle = True)

# Flow validation images in batches of 20 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                        target_size = (150, 150),
                                                        batch_size =20,
                                                        class_mode = 'binary',
                                                        shuffle = False)
```

#### Model Fit
```
callbacks = myCallback()
history = model.fit(
    training_dataset,
    validation_data = validation_dataset,
    epochs = 10,
    verbose = 2,
    callbacks = [myCallback()],
)
```

#### Plotting Training VS Validation accuracy & loss

```
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Name: Vijay Shankar M           Register Number:  212222040178    ')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Name: Vijay Shankar M           Register Number:  212222040178   ')
plt.title('Training and validation Loss')
plt.legend(loc=0)
plt.figure()


plt.show()



```


## OUTPUT
### Training Accuracy, Validation Accuracy Vs Iteration Plot
![Screenshot 2024-10-28 232745](https://github.com/user-attachments/assets/eca3539d-8c88-4fa1-9971-5f8422a2e39e)

</br>
</br>
</br>

### Training Loss, Validation Loss Vs Iteration Plot
![Screenshot 2024-10-28 232801](https://github.com/user-attachments/assets/9411010b-fdae-4c02-bbff-77fe7821bc27)

</br>
</br>
</br>

### Conclusion
<br>
The final model successfully learned to differentiate between horse and human images with high accuracy. By employing transfer learning with InceptionV3, we could take advantage of the pre-trained features from a larger dataset, significantly improving the training efficiency. The model was able to stop training automatically once 97% accuracy was achieved, indicating its robustness. Data augmentation techniques also helped in increasing the generalizability of the model, as evidenced by the minimal difference between the training and validation performance.

## RESULT
</br>
The model achieved over 97% accuracy in classifying horse and human images using transfer learning with the InceptionV3 architecture. Data augmentation and early stopping ensured effective training and prevented overfitting.
</br>
