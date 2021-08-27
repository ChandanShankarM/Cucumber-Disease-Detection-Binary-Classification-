from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import os
import numpy as np
from PIL import Image

# Construct an instance of the ImageDataGenerator class
# Pass the augmentation parameters through the constructor. 

datagen = ImageDataGenerator(
        rotation_range=45,     #Random rotation between 0 and 45
        width_shift_range=0.2,   #% shift
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='constant', cval=125)    #Also try nearest, constant, reflect, wrap
image_directory = r'E:\hindi\Not_a_cucumber leaf'
SIZE = 128
dataset = []

my_images = os.listdir(image_directory)
for i, image_name in enumerate(my_images):
    if (image_name.split('.')[1] == 'jpg'):
        image = io.imread(image_directory + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE,SIZE))
        dataset.append(np.array(image))

x = np.array(dataset)
i = 0
for batch in datagen.flow(x, batch_size=16,  
                          save_to_dir=r'E:\hindi\augumented_not', 
                          save_prefix='aug', 
                          save_format='jpg'):
    i += 1
    if i > 20:
        break