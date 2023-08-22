import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import pathlib 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

class Classifier():
  def __init__(self, data_path):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
      except RuntimeError as e:
        print(e)

    data_dir = data_path
    data_dir = pathlib.Path(data_dir).with_suffix('')

    self.batch_size = 32
    self.img_height = 64
    self.img_width = 64

    self.train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(self.img_height, self.img_width),
        batch_size=self.batch_size)

    self.val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(self.img_height, self.img_width),
        batch_size=self.batch_size)

    class_names = self.train_ds.class_names

    normalization_layer = layers.Rescaling(1./255)

    normalized_ds = self.train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    #first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    #print(np.min(first_image), np.max(first_image))

    data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                        input_shape=(self.img_height,
                                    self.img_width,
                                    3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
    )

    num_classes = len(class_names)

    self.model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, name="outputs")
    ])



    self.model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  
  def train(self):
    epochs=20

    history = self.model.fit(
    self.train_ds,
    validation_data=self.val_ds,
    epochs=epochs
    )
    return
  
  def predict(self,image):
    img = tf.expand_dims(image, 0) # Create a batch
    predictions = self.model.predict(img)
    score = tf.nn.softmax(predictions[0]).numpy()
    max_score = np.max(score)
    return float(np.argmax(score) and max_score > 0.99)

if __name__ == "__main__":
  #train classifier and use it to convert rewards in dataset
  path = "/home/rad/.keras/franka_gripper/"
  classifier = Classifier(path)
  classifier.train()

  import numpy as np
  from os import listdir
  from os.path import isfile, join

  mypath = '/home/rad/Documents/ClothHangProject/episodes/rgbd_new_2/'
  new_path = '/home/rad/Documents/ClothHangProject/episodes/new_reward_fixed_2/'
  onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
  for file in onlyfiles:
    episode = np.load(join(mypath, file))
    observations = episode['observation']
    images = episode['image']
    depths = episode['depth']
    rewards = []
    is_firsts = episode['is_first']
    is_lasts = episode['is_last']
    is_terminals = episode['is_terminal']
    actions = episode['action']

    for i in range(len(episode['image'])):
      image = episode['image'][i]
      cls_prediction = classifier.predict(image)
      rewards.append(cls_prediction)
    filename = join(new_path, file)
    np.savez_compressed(filename, observation=observations, image=images, depth=depths, reward=rewards, is_first=is_firsts, is_last=is_lasts, is_terminal=is_lasts,action=actions, )


